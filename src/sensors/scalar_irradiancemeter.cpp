#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-ScalarIrradianceMeter:

Scalar irradiance meter (:monosp:`ScalarIrradianceMeter`)
--------------------------------------------

.. pluginparameters::
 * - to_world
   - |transform|
   - Specifies an optional camera-to-world transformation.
     (Default: none (i.e. camera space = world space))

 * - origin
   - |point|
   - Location from which the sensor will be recording in world coordinates.
     Must be used with `direction`.

 * - direction
   - |vector|
   - Alternative (and exclusive) to `to_world`. Direction in which the
     sensor is pointing in world coordinates. Must be used with `origin`.

 * - fov
    - |float|
    - Angle around the normal being measured. At 180 degrees, the sensor
      measures all directions, at 90, the upper hemisphere. (Default: 180)
      
 * - srf
   - |spectrum|
   - Sensor Response Function that defines the :ref:`spectral sensitivity <explanation_srf_sensor>`
     of the sensor (Default: :monosp:`none`)

This sensor plugin implements a delta irradiance meter, which measures
the incident power per unit area at a single point by integrating the radiance around its normal.
Scalar irradiance meters respond equally to all radiance directions, unlike vector (also called plane) irradiance meters.
This mimics real spherical collectors.

This sensor is used with films of 1 by 1 pixels.

By default it integrates the radiance from all directions, but the angle can be
changed by providing a custom :monosp:`max_cosine` value (default: 180 deg).


.. tabs::
    .. code-tab:: xml
        :name: delta-irradiancemeter

        <sensor type="ScalarIrradianceMeter">
            <!-- film -->
        </sensor>

    .. code-tab:: python

        'type': 'sphere',
        'sensor': {
            'type': 'ScalarIrradianceMeter'
            'film': {
                # ...
            }
        }
*/

MI_VARIANT class ScalarIrradianceMeter final : public Sensor<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Sensor, m_film, m_shape, sample_wavelengths)
    MI_IMPORT_TYPES(Shape)

    ScalarIrradianceMeter(const Properties &props) : Base(props) {
        if (props.has_property("to_world")) {
            // if direction and origin are present but overridden by
            // to_world, they must still be marked as queried
            props.mark_queried("direction");
            props.mark_queried("origin");
        } else {
            if (props.has_property("direction") !=
                props.has_property("origin")) {
                Throw("If the sensor is specified through origin and direction "
                      "both values must be set!");
            }

            if (props.has_property("direction")) {
                ScalarPoint3f origin     = props.get<ScalarPoint3f>("origin");
                ScalarVector3f direction = props.get<ScalarVector3f>("direction");
                ScalarPoint3f target     = origin + direction;
                auto [up, unused]        = coordinate_system(dr::normalize(direction));

                m_to_world = ScalarTransform4f::look_at(origin, target, up);
                dr::make_opaque(m_to_world);
            }
        }

        if (dr::all(m_film->size() != ScalarPoint2i(1, 1)))
            Throw("This sensor only supports films of size 1x1 Pixels!");

        if (m_film->rfilter()->radius() >
            0.5f + math::RayEpsilon<Float>)
            Log(Warn, "This sensor should be used with a reconstruction filter "
                      "with a radius of 0.5 or lower (e.g. default box)");

        ScalarFloat cutoff_angle = props.get<ScalarFloat>("cutoff_angle", 20.0f);
        m_beam_width   = props.get<ScalarFloat>("beam_width", cutoff_angle * 3.0f / 4.0f);
        m_cutoff_angle = dr::deg_to_rad(cutoff_angle);
        m_beam_width   = dr::deg_to_rad(m_beam_width);
        m_inv_transition_width = 1.0f / (m_cutoff_angle - m_beam_width);
        m_cos_cutoff_angle = dr::cos(m_cutoff_angle);
        m_cos_beam_width   = dr::cos(m_beam_width);
        Assert(dr::all(m_cutoff_angle >= m_beam_width));
        m_uv_factor = dr::tan(m_cutoff_angle);

        dr::make_opaque(m_beam_width, m_cutoff_angle, m_uv_factor,
                        m_cos_beam_width, m_cos_cutoff_angle,
                        m_inv_transition_width);

        m_needs_sample_2 = false;
        m_needs_sample_3 = false;
        
    }


    
    std::pair<RayDifferential3f, Spectrum>
    sample_ray_differential(Float time, Float wavelength_sample,
                            const Point2f & /*position_sample*/,
                            const Point2f & /*aperture_sample*/,
                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        RayDifferential3f ray = dr::zeros<RayDifferential3f>();
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
                               wavelength_sample,
                               active);
        ray.wavelengths = wavelengths;

        // 2. Set ray origin and direction
        ray.o = m_to_world.value().transform_affine(Point3f(0.f, 0.f, 0.f));
        ray.d = m_to_world.value().transform_affine(Vector3f(0.f, 0.f, 1.f));
        ray.o += ray.d * math::RayEpsilon<Float>;

        // 3. Set differentials; since the film size is always 1x1, we don't
        //    have differentials
        ray.has_differentials = false;

        return { ray, wav_weight };
    }

    ScalarBoundingBox3f bbox() const override {
        // Return an invalid bounding box
        return ScalarBoundingBox3f();
    }

    std::pair<RayDifferential3f, Spectrum>
    sample_ray_differential(Float time, Float wavelength_sample,
                            const Point2f & sample2,
                            const Point2f & sample3,
                            Mask active) const override {

        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample spatial component
        PositionSample3f ps = m_shape->sample_position(time, sample2, active);

        // 2. Sample directional component
        Vector3f local = warp::square_to_cosine_hemisphere(sample3);

        // 3. Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
                               wavelength_sample,
                               active);

        Vector3f d = Frame3f(ps.n).to_world(local);
        Point3f o = ps.p + d * math::RayEpsilon<Float>;

        return {
            RayDifferential3f(o, d, time, wavelengths),
            depolarizer<Spectrum>(wav_weight) * dr::Pi<ScalarFloat>
        };
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
        return { m_shape->sample_direction(it, sample, active), dr::Pi<ScalarFloat> };
    }

    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask active) const override {
        return m_shape->pdf_direction(it, ds, active);
    }

    Spectrum eval(const SurfaceInteraction3f &/*si*/, Mask /*active*/) const override {
        return dr::Pi<ScalarFloat> / m_shape->surface_area();
    }

    ScalarBoundingBox3f bbox() const override { return m_shape->bbox(); }

    std::string to_string() const override {
        using string::indent;

        std::ostringstream oss;
        oss << "ScalarIrradianceMeter[" << std::endl << "  surface_area = ";

        if (m_shape)
            oss << m_shape->surface_area();
        else
            oss << " <no shape attached!>";
        oss << "," << std::endl;

        oss << "  film = " << indent(m_film) << "," << std::endl << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(ScalarIrradianceMeter, Sensor)
MI_EXPORT_PLUGIN(ScalarIrradianceMeter, "ScalarIrradianceMeter");
NAMESPACE_END(mitsuba)
