
#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/distr_1d.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-irregular-tabphase:

Lookup table phase function (:monosp:`tabphase`)
------------------------------------------------

.. pluginparameters::


 * - cosines
   - |string|
   - Cosine values where the function is defined.
   - |exposed|, |differentiable|

 * - values
   - |string|
   - Values of the phase function at the specified cosines.
   - |exposed|, |differentiable|, |discontinuous|

This spectrum returns linearly interpolated phase function values from *irregularly*
placed cosine samples.

This plugin implements a generic phase function model for isotropic media
parametrized by a lookup table giving values of the phase function as a
function of the cosine of the scattering angle.

.. admonition:: Notes

   * The scattering angle cosine is here defined as the dot product of the
     incoming and outgoing directions, where the incoming, resp. outgoing
     direction points *toward*, resp. *outward* the interaction point.
   * From this follows that :math:`\cos \theta = 1` corresponds to forward
     scattering.
   * Lookup table points are regularly spaced between -1 and 1.
   * Phase function values are automatically normalized.
*/


// TODO: Finish the 

template <typename Float, typename Spectrum>
class IrregularTabulatedPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext)

public:
    IrregularTabulatedPhaseFunction(const Properties &props) : Base(props) {
        if (props.type("values") == Properties::Type::String) {
            std::vector<std::string> wavelengths_str =
                string::tokenize(props.string("wavelengths"), " ,");
            std::vector<std::string> entry_str, values_str =
                string::tokenize(props.string("values"), " ,");

            if (values_str.size() != wavelengths_str.size())
                Throw("IrregularSpectrum: 'wavelengths' and 'values' parameters must have the same size!");

            std::vector<ScalarFloat> values, wavelengths;
            values.reserve(values_str.size());
            wavelengths.reserve(values_str.size());

            for (size_t i = 0; i < values_str.size(); ++i) {
                try {
                    wavelengths.push_back(string::stof<ScalarFloat>(wavelengths_str[i]));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'", wavelengths_str[i]);
                }
                try {
                    values.push_back(string::stof<ScalarFloat>(values_str[i]));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'", values_str[i]);
                }
            }

            m_distr = IrregularContinuousDistribution<Wavelength>(
                wavelengths.data(), values.data(), values.size()
            );
        } else {
            // Scene/property parsing is in double precision, cast to single precision depending on variant.
            size_t size = props.get<size_t>("size");
            const double *whl = static_cast<const double*>(props.pointer("wavelengths"));
            const double *ptr = static_cast<const double*>(props.pointer("values"));

            if constexpr (std::is_same_v<ScalarFloat, double>) {
                m_distr = IrregularContinuousDistribution<Wavelength>(whl, ptr, size);
            } else {
                std::vector<ScalarFloat> values(size), wavelengths(size);
                for (size_t i=0; i < size; ++i) {
                    values[i] = (ScalarFloat) ptr[i];
                    wavelengths[i] = (ScalarFloat) whl[i];
                }
                m_distr = IrregularContinuousDistribution<Wavelength>(
                    wavelengths.data(), values.data(), size);
            }
        }
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("cosines", m_distr.nodes(), +ParamFlags::Differentiable);
        callback->put_parameter("values",      m_distr.pdf(),   +ParamFlags::Differentiable);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        m_distr.update();
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        if constexpr (is_spectral_v<Spectrum>)
            return m_distr.eval_pdf(si.wavelengths, active);
        else {
            DRJIT_MARK_USED(si);
            NotImplementedError("eval");
        }
    }

    Wavelength pdf_spectrum(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        if constexpr (is_spectral_v<Spectrum>)
            return m_distr.eval_pdf_normalized(si.wavelengths, active);
        else {
            DRJIT_MARK_USED(si);
            NotImplementedError("pdf");
        }
    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext & /* ctx */,
                                                 const MediumInteraction3f &mi,
                                                 Float /* sample1 */,
                                                 const Point2f &sample2,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        // TODO: THIS must be very similar to tabphase, the following is just from irregular.cpp:
        if constexpr (is_spectral_v<Spectrum>)
            return { m_distr.sample(sample, active), m_distr.integral() };
        else {
            DRJIT_MARK_USED(sample);
            NotImplementedError("sample");
        }
    }

    Float mean() const override {
        ScalarVector2f range = m_distr.range();
        return m_distr.integral() / (range[1] - range[0]);
    }

    ScalarVector2f wavelength_range() const override {
        return m_distr.range();
    }

    ScalarFloat spectral_resolution() const override {
        return m_distr.interval_resolution();
    }

    ScalarFloat max() const override {
        return m_distr.max();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "IrregularSpectrum[" << std::endl
            << "  distr = " << string::indent(m_distr) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    IrregularContinuousDistribution<Wavelength> m_distr;
};

MI_IMPLEMENT_CLASS_VARIANT(IrregularSpectrum, Texture)
MI_EXPORT_PLUGIN(IrregularSpectrum, "Irregular interpolated tabulated phase function")
NAMESPACE_END(mitsuba)
