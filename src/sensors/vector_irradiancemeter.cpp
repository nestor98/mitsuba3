#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-VectorIrradianceMeter:

Irradiance meter (:monosp:`VectorIrradianceMeter`)
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

 * - srf
   - |spectrum|
   - Sensor Response Function that defines the :ref:`spectral sensitivity <explanation_srf_sensor>`
     of the sensor (Default: :monosp:`none`)

This sensor plugin implements a delta irradiance meter, which measures
the incident power per unit area at a single point by integrating the radiance around its normal,
wheighted by the cosine of the angle between the normal and the radiance direction.
This mimics plane irradiance meters commonly used in ocean optics.

This sensor is used with films of 1 by 1 pixels.

By default it integrates the radiance from its upper hemisphere, but the angle can be
changed by providing a custom :monosp:`max_cosine` value (default: 90 deg).


.. tabs::
    .. code-tab:: xml
        :name: delta-irradiancemeter

        <sensor type="VectorIrradianceMeter">
            <!-- film -->
        </sensor>

    .. code-tab:: python

        'type': 'sphere',
        'sensor': {
            'type': 'VectorIrradianceMeter'
            'film': {
                # ...
            }
        }
*/

MI_VARIANT class VectorIrradianceMeter final : public Sensor<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Sensor, m_film, m_shape, sample_wavelengths)
    MI_IMPORT_TYPES(Shape)

    VectorIrradianceMeter(const Properties &props) : Base(props) {
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



        m_needs_sample_2 = false;
        m_needs_sample_3 = false;
        
    }


    
    std::pair<RayDifferential3f, Spectrum>
    sample_ray_differential(Float time, Float wavelength_sample,
                            const Point2f & sample2,
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

        // 2. Sample local direction (cos weighted)
        ray.d = warp::square_to_cosine_hemisphere(sample2);

        // 3. Set ray origin and direction in world space:
        ray.o = m_to_world.value().transform_affine(Point3f(0.f, 0.f, 0.f));
        ray.d = m_to_world.value().transform_affine(ray.d);
        ray.o += ray.d * math::RayEpsilon<Float>;

        // 4. Set differentials; since the film size is always 1x1, we don't
        //    have differentials
        ray.has_differentials = false;

        return { 
            ray, 
            depolarizer<Spectrum>(wav_weight) * dr::Pi<ScalarFloat> 
        };
        // Note: the depolarizer here means that this sensor is not polarization sensitive
        // this would need to be handled differently if polarization sensitivity is desired
    }

    ScalarBoundingBox3f bbox() const override {
        // Return an invalid bounding box
        return ScalarBoundingBox3f();
    }

    // sample_direction functions??? 
    // std::pair<DirectionSample3f, Spectrum>
    // sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
    //     return { m_shape->sample_direction(it, sample, active), dr::Pi<ScalarFloat> };
    // }

    // Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
    //                     Mask active) const override {
    //     return m_shape->pdf_direction(it, ds, active);
    // }

    // No eval, as the sensor only occupies a delta position
    // Spectrum eval(const SurfaceInteraction3f &/*si*/, Mask /*active*/) const override {
    //     return dr::Pi<ScalarFloat> / m_shape->surface_area();
    // }


    std::string to_string() const override {
        using string::indent;
        std::ostringstream oss;
        oss << "VectorIrradianceMeter[" << std::endl
            << "  to_world = " << m_to_world << "," << std::endl
            << "  film = " << indent(m_film) << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(VectorIrradianceMeter, Sensor)
MI_EXPORT_PLUGIN(VectorIrradianceMeter, "vectorirradiancemeter");
NAMESPACE_END(mitsuba)
