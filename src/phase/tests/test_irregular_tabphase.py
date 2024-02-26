import pytest
import drjit as dr
import mitsuba as mi
import numpy as np

from itertools import product


def test_create(variant_scalar_rgb):
    p = mi.load_dict({
        "type" : "irregular_tabphase",
        "cosines" : "-0.8, 0.1, 0.85",
        "values" : "0.5, 1.0, 1.5"
    })
    assert p is not None

# it would be nice to have the same interface as the irregular spectrum type, then we could instantiate it like this function
# . However, this would require a bunch of changes
# to xml.cpp and python/xml_v.cpp. For now, we just use a string to specify the values.
# def test_create_lists(variant_scalar_rgb, x=np.linspace(-1,1,10), y=np.random.rand(10)):
#     # make sure the "pythonic" init works
#     p = mi.load_dict({"type" : "irregular_tabphase", "values" : list(zip(x, y))})
#     # p = mi.load_dict({"type" : "irregular_tabphase", "values" : list(zip(np.linspace(-1,1,10), np.random.rand(10)))})

#     assert p is not None


def test_eval(variant_scalar_rgb):
    """
    Compare eval() output with a reference implementation written in Python.
    We make sure that the values we use to initialize the plugin are such that
    the phase function has an asymmetric lobe.
    """
    import numpy as np

    # Phase function table definition
    ref_y = np.array([0.5, 1.0, 1.5])
    ref_x = np.linspace(-1, 1, len(ref_y))
    ref_integral = np.trapz(ref_y, ref_x)

    def eval(wi, wo):
        # Python implementation used as a reference
        wi = wi.reshape((-1, 3))
        wo = wo.reshape((-1, 3))

        if wi.shape[0] == 1:
            wi = np.broadcast_to(wi, wo.shape)
        if wo.shape[0] == 1:
            wo = np.broadcast_to(wo, wi.shape)

        cos_theta = np.array([np.dot(a, b) for a, b in zip(wi, wo)])
        return 0.5 / np.pi * np.interp(-cos_theta, ref_x, ref_y) / ref_integral

    wi = np.array([[0, 0, -1]])
    thetas = np.linspace(0, np.pi / 2, 16)
    phis = np.linspace(0, np.pi, 16)
    wos = np.array(
        [
            (
                dr.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            )
            for theta, phi in product(thetas, phis)
        ]
    )
    ref_eval = eval(wi, wos)

    # Evaluate Mitsuba implementation
    tab = mi.load_dict({"type": "irregular_tabphase", 
                        "cosines": ", ".join([str(x) for x in ref_x]),
                        "values": ", ".join([str(y) for y in ref_y])})
    ctx = mi.PhaseFunctionContext(None)
    mei = mi.MediumInteraction3f()
    mei.wi = wi
    tab_eval = np.zeros_like(ref_eval)
    for i, wo in enumerate(wos):
        tab_eval[i] = tab.eval_pdf(ctx, mei, wo)[1]

    # Compare reference and plugin outputs
    assert np.allclose(ref_eval, tab_eval)


def test_sample(variant_scalar_rgb):
    """
    Check if the sampling routine uses consistent incoming-outgoing orientation
    conventions.
    """

    tab = mi.load_dict({"type": "irregular_tabphase", "cosines": "-1, 0, 1.0", "values": "0.0, 0.5, 1.0"})
    ctx = mi.PhaseFunctionContext(None)
    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    # The passed sample corresponds to forward scattering
    wo, w, pdf = tab.sample(ctx, mei, 0, (1, 0))

    # The sampled direction indicates forward scattering in the "graphics"
    # convention
    assert dr.allclose(wo, [0, 0, -1])

    # The expected value was derived manually from the PDF expression.
    # An incorrect convention (i.e. using -cos θ to fetch the PDF value) will
    # yield 0 thanks to the values used to initialize the distribution and will
    # make the test fail.
    assert dr.allclose(pdf, 0.5 / dr.pi)


def test_chi2(variants_vec_backends_once_rgb):
    sample_func, pdf_func = mi.chi2.PhaseFunctionAdapter(
        "irregular_tabphase", " <string name='cosines' value='-1, 0, 1.0'/> <string name='values' value='0.5, 1.0, 1.5'/> "
    )

    chi2 = mi.chi2.ChiSquareTest(
        domain=mi.chi2.SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=3,
    )

    result = chi2.run()
    # chi2._dump_tables()
    assert result


def test_traverse(variant_scalar_rgb):
    # Phase function table definition
    import numpy as np

    ref_vals = [0.5, 0.75, 1.0, 1.25, 1.5]
    ref_y = np.array(ref_vals)
    ref_x = np.linspace(-1, 1, len(ref_y))
    ref_integral = np.trapz(ref_y, ref_x)

    # Initialise as isotropic and update with parameters
    phase = mi.load_dict({"type": "irregular_tabphase", "cosines" : "-1, -0.5, 0, 0.5, 1", "values": "1, 1, 1, 1, 1"})
    params = mi.traverse(phase)
    params["values"] = ref_vals
    params.update()

    # Distribution parameters are updated
    params = mi.traverse(phase)
    assert dr.allclose(params["values"], ref_vals)

    # The plugin itself evaluates consistently
    ctx = mi.PhaseFunctionContext(None)
    mei = mi.MediumInteraction3f()
    mei.wi = np.array([0, 0, -1])
    wo = [0, 0, 1]
    assert dr.allclose(phase.eval_pdf(ctx, mei, wo)[0], dr.inv_two_pi * 1.5 / ref_integral)
