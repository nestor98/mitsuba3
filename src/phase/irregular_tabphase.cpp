#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/phase.h>

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

template <typename Float, typename Spectrum>
class IrregularTabulatedPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext)

public:
    IrregularTabulatedPhaseFunction(const Properties &props) : Base(props) {
        if (props.type("values") == Properties::Type::String) {
            std::vector<std::string> cosines_str =
                string::tokenize(props.string("cosines"), " ,");
            std::vector<std::string> entry_str, values_str =
                string::tokenize(props.string("values"), " ,");

            if (values_str.size() != cosines_str.size())
                Throw("IrregularTabulatedPhaseFunction: 'cosines' and 'values' parameters must have the same size!");

            std::vector<ScalarFloat> values, cosines;
            values.reserve(values_str.size());
            cosines.reserve(values_str.size());

            for (size_t i = 0; i < values_str.size(); ++i) {
                try {
                    cosines.push_back(string::stof<ScalarFloat>(cosines_str[i]));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'", cosines_str[i]);
                }
                try {
                    values.push_back(string::stof<ScalarFloat>(values_str[i]));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'", values_str[i]);
                }
            }

            m_distr = IrregularContinuousDistribution<Float>(
                cosines.data(), values.data(), values.size()
            );
        } else {
            // Scene/property parsing is in double precision, cast to single precision depending on variant.
            size_t size = props.get<size_t>("size");
            const double *cs  = static_cast<const double*>(props.pointer("cosines"));
            const double *ptr = static_cast<const double*>(props.pointer("values"));

            if constexpr (std::is_same_v<ScalarFloat, double>) {
                m_distr = IrregularContinuousDistribution<Float>(cs, ptr, size);
            } else {
                std::vector<ScalarFloat> values(size), cosines(size);
                for (size_t i=0; i < size; ++i) {
                    values[i] = (ScalarFloat) ptr[i];
                    cosines[i] = (ScalarFloat) cs[i];
                }
                m_distr = IrregularContinuousDistribution<Float>(
                    cosines.data(), values.data(), size);
            }
        }
        
        m_flags = +PhaseFunctionFlags::Anisotropic;
        dr::set_attr(this, "flags", m_flags);
        m_components.push_back(m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("cosines", m_distr.nodes(), +ParamFlags::Differentiable);
        callback->put_parameter("values",      m_distr.pdf(),   +ParamFlags::Differentiable);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        m_distr.update();
    }
    
    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext & /* ctx */,
                                                 const MediumInteraction3f &mi,
                                                 Float /* sample1 */,
                                                 const Point2f &sample2,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        // Sample a direction in physics convention.
        // We sample cos θ' = cos(π - θ) = -cos θ.
        Float cos_theta_prime = m_distr.sample(sample2.x());
        Float sin_theta_prime =
            dr::safe_sqrt(1.f - cos_theta_prime * cos_theta_prime);
        auto [sin_phi, cos_phi] =
            dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());
        Vector3f wo{ sin_theta_prime * cos_phi, sin_theta_prime * sin_phi,
                     cos_theta_prime };

        // Switch the sampled direction to graphics convention and transform the
        // computed direction to world coordinates
        wo = -mi.to_world(wo);

        // Retrieve the PDF value from the physics convention-sampled angle
        Float pdf = m_distr.eval_pdf_normalized(cos_theta_prime, active) *
                    dr::InvTwoPi<ScalarFloat>;

        return { wo, 1.f, pdf };
    }


    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext & /* ctx */,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        // The data is laid out in physics convention
        // (with cos θ = 1 corresponding to forward scattering).
        // This parameterization differs from the convention used internally by
        // Mitsuba and is the reason for the minus sign below.
        Float cos_theta = -dot(wo, mi.wi);
        Float pdf = m_distr.eval_pdf_normalized(cos_theta, active) * dr::InvTwoPi<ScalarFloat>;
        return { pdf, pdf };
    }


    std::string to_string() const override {
        std::ostringstream oss;
        oss << "IrregularTabulatedPhaseFunction[" << std::endl
            << "  distr = " << string::indent(m_distr) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    IrregularContinuousDistribution<Float> m_distr;
};

MI_IMPLEMENT_CLASS_VARIANT(IrregularTabulatedPhaseFunction, PhaseFunction)
MI_EXPORT_PLUGIN(IrregularTabulatedPhaseFunction, "Irregular interpolated tabulated phase function")
NAMESPACE_END(mitsuba)
