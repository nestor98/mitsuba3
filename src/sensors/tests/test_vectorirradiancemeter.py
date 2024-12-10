import pytest
import drjit as dr
import mitsuba as mi

import numpy as np

## Constructing deltairradiancemeters is identical to radiancemeters, so same tests:
def make_sensor(origin=None, direction=None, to_world=None, pixels=1, radiancemeter=False):
    d = {
        "type": "vectorirradiancemeter",
        "film": {
            "type": "hdrfilm",
            "width": pixels,
            "height": pixels,
            "rfilter": {"type": "box"}
        }
    }

    if origin is not None:
        d["origin"] = origin
    if direction is not None:
        d["direction"] = direction
    if to_world is not None:
        d["to_world"] = to_world
    if radiancemeter:
        d["type"] = "radiancemeter"

    return mi.load_dict(d)


def test_construct(variant_scalar_rgb):
    # Test construct from to_world
    sensor = make_sensor(to_world=mi.ScalarTransform4f().look_at(
        origin=[0, 0, 0],
        target=[0, 1, 0],
        up=[0, 0, 1]
    ))
    assert not sensor.bbox().valid()  # Degenerate bounding box
    assert dr.allclose(
        sensor.world_transform().matrix,
        [[-1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]]
    )

    # Test construct from origin and direction
    sensor = make_sensor(origin=[0, 0, 0], direction=[0, 1, 0])
    assert not sensor.bbox().valid()  # Degenerate bounding box
    assert dr.allclose(
        sensor.world_transform().matrix,
        [[0, 1, 0, 0],
         [0, 0, 1, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 1]]
    )

    # Test to_world overriding direction + origin
    sensor = make_sensor(
        to_world=mi.ScalarTransform4f().look_at(
            origin=[0, 0, 0],
            target=[0, 1, 0],
            up=[0, 0, 1]
        ),
        origin=[1, 0, 0],
        direction=[4, 1, 0]
    )
    assert not sensor.bbox().valid()  # Degenerate bounding box
    assert dr.allclose(
        sensor.world_transform().matrix,
        [[-1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]]
    )

    # Test raise on missing direction or origin
    with pytest.raises(RuntimeError):
        sensor = make_sensor(direction=[0, 1, 0])

    with pytest.raises(RuntimeError):
        sensor = make_sensor(origin=[0, 1, 0])

    # Test raise on wrong film size
    with pytest.raises(RuntimeError):
        sensor = make_sensor(pixels=2)




@pytest.mark.parametrize(
    ("origin", "direction"),
    [([2.0, 5.0, 8.3], [0.0, 1.0, 0.0]), ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0]), ([1.0, 4.0, 0.0], [1.0, 4.0, 0.0])]
)
def test_sampling(variant_scalar_rgb, origin, direction, np_rng):
    """
    We construct a vector irradiancemeter at some origins and directions and test that
    sampled rays point to the hemisphere around the direction
    """
    o, d = mi.ScalarVector3f(origin), mi.ScalarVector3f(direction)

    sensor = make_sensor(origin=o, direction=d)

    num_samples = 100

    wav_samples = np_rng.random((num_samples,))
    pos_samples = np_rng.random((num_samples, 2))
    dir_samples = np_rng.random((num_samples, 2))

    for i in range(num_samples):
        ray = sensor.sample_ray_differential(
            0.0, wav_samples[i], pos_samples[i], dir_samples[i])[0]

        # assert that the ray starts at the origin
        assert dr.allclose(ray.o, o, atol=1e-4)
        # assert that all rays point away from the direction
        assert dr.dot(direction, ray.d) > 0.0


def constant_emitter_dict(radiance):
    return {
        "type": "constant",
        "radiance": {"type": "uniform", "value": radiance}
    }


@pytest.mark.parametrize("radiance", [2.04, 1.0, 0.0])
def test_incoming_flux(variant_scalar_rgb, radiance, np_rng):
    """
    We test the recorded power density of the vector irradiance meter, by creating a simple scene:
    The irradiance meter is at the coordinate origin
    surrounded by a constant environment emitter.
    We sample a number of rays and average their contribution to the cumulative power
    density.
    We expect the average value to be \\pi * L with L the radiance of the constant
    emitter.
    """

    scene_dict = {
        'type': 'scene',
        'sensor': make_sensor(origin=[0,0,0], direction=[0,0,1]),
        'emitter': constant_emitter_dict(radiance)
    }

    scene = mi.load_dict(scene_dict)
    sensor = scene.sensors()[0]

    power_density_cum = 0.0
    num_samples = 100

    wav_samples = np_rng.random((num_samples,))
    pos_samples = np_rng.random((num_samples, 2))
    dir_samples = np_rng.random((num_samples, 2))

    for i in range(num_samples):
        ray, weight = sensor.sample_ray_differential(
            0.0, wav_samples[i], pos_samples[i], dir_samples[i])

        intersection = scene.ray_intersect(ray)
        power_density_cum += weight * \
            intersection.emitter(scene).eval(intersection)
    power_density_avg = power_density_cum / float(num_samples)

    assert dr.allclose(power_density_avg, mi.Spectrum(dr.pi * radiance))


@pytest.mark.parametrize("radiance", [2.04, 1.0, 0.0])
def test_incoming_flux_integrator(variant_scalar_rgb, radiance):
    """
    We test the recorded power density of the vector irradiance meter, by creating a simple scene:
    The irradiance meter is at the coordinate origin
    surrounded by a constant environment emitter.
    We render the scene with the path tracer integrator and compare the recorded  power
    density with our theoretical expectation.
    We expect the average value to be \\pi * L with L the radiance of the constant
    emitter.
    """

    scene_dict = {
        'type': 'scene',
        'sensor': make_sensor(origin=[0,0,0], direction=[0,0,1]),
        'emitter': constant_emitter_dict(radiance),
        'integrator': {'type': 'path'}
    }

    scene = mi.load_dict(scene_dict)
    scene.integrator().render(scene, seed=0)
    film = scene.sensors()[0].film()

    img = film.bitmap(raw=True).convert(mi.Bitmap.PixelFormat.Y,
                                        mi.Struct.Type.Float32, srgb_gamma=False)

    assert dr.allclose(mi.TensorXf(img), (radiance * dr.pi))


def get_envmap(radiance):
    import numpy as np
    theta = np.linspace(0, np.pi, 11)
    im = np.ones((11, 11)) * radiance * np.maximum(np.cos(theta), 0.0)
    im = dr.maximum(0.0, mi.TensorXf(im.T))

    bmp = mi.Bitmap(im)
    
    bmp.write("cos-envmap-full.exr")

    return {
        "type": "envmap",
        "bitmap": mi.Bitmap(im)
    }


# @pytest.mark.parametrize("radiance", [2.04, 1.0, 100.0])
# def test_my_envmap(radiance):

#     scene_dict = {
#         'type': 'scene',
#         'sensor': {"type": "perspective", "fov":90, 
#                     "to_world": mi.ScalarTransform4f().look_at(target=[0,1,0], origin=[0,0,0], up=[0,1,0]),
#                     "film": {
#                         "type": "hdrfilm",
#                         "width": 32,
#                         "height": 32,
#                         "rfilter": {"type": "box"}
#                     }},
#         'emitter': get_envmap(radiance),
#         'integrator': {'type': 'path'}
#     }
#     scene = mi.load_dict(scene_dict)
#     img = mi.render(scene, seed=0, spp=100)
#     # scene.integrator().render(scene, seed=0)
#     # film = scene.sensors()[0].film()

#     # img = film.bitmap()
#                 # .convert(mi.Bitmap.PixelFormat.Y,
#                 #                         mi.Struct.Type.Float32, srgb_gamma=False)

#     img.write("cos-envmap.exr")


@pytest.mark.parametrize(
    ("radiance", "angle"),
    [(2.04, 90), (1.0, 0.0), (1000.0, 30.0)]
)
def test_cosine_weight(variants_any_llvm, radiance, angle):
    """
    We test the recorded power density of the vector irradiance meter, by creating a simple scene:
    The irradiance meter is at the coordinate origin looking towards the given direction,
    surrounded by an envmap with a cosine weighted radiance (L(theta)=cos(theta)).

    We render the scene with the path tracer integrator and compare the recorded  power
    density with our theoretical expectation.
    We expect the average value to be \\pi/4 * L (pi/4 is the integral of cos^2 for the hemisphere),
    with L being the base radiance <radiance>
    """

    scene_dict = {
        'type': 'scene',
        'sensor': make_sensor(origin=[0,0,0], direction=[0,1,0]),
        'emitter': get_envmap(radiance),
        'integrator': {'type': 'path'}
    }

    scene = mi.load_dict(scene_dict)
    scene.integrator().render(scene, seed=0, spp=3200)
    film = scene.sensors()[0].film()

    img = film.bitmap(raw=True).convert(mi.Bitmap.PixelFormat.Y,
                                        mi.Struct.Type.Float32, srgb_gamma=False)

    assert dr.allclose(mi.TensorXf(img), (radiance * dr.pi/4.0 )), "Measured: " + str(dr.ravel(mi.TensorXf(img))) + " Theory: " + str(radiance * dr.pi/4.0 )


# def test_shape_accessors(variants_vec_rgb):
#     center_v = mi.ScalarVector3f(0.0)
#     radius = 1.0
#     shape = mi.load_dict(sensor_shape_dict(radius, center_v))
#     shape_ptr = mi.ShapePtr(shape)

#     assert type(shape.sensor()) == mi.Sensor
#     assert type(shape_ptr.sensor()) == mi.SensorPtr

#     sensor = shape.sensor()
#     sensor_ptr = mi.SensorPtr(sensor)

#     assert type(sensor.get_shape()) == mi.Shape
#     assert type(sensor_ptr.get_shape()) == mi.ShapePtr

@pytest.mark.parametrize(
    ("radiance", "angle"),
    [(2.04, 90), (1.0, 0.0), (1000.0, 30.0)]
)
def test_cosine_weight_v2(variants_any_llvm, radiance, angle):
    """
    We test the recorded power density of the vector irradiance meter, by creating a simple scene:
    The irradiance meter is at the coordinate origin looking towards the given direction,
    surrounded by an envmap with a cosine weighted radiance (L(theta)=cos(theta)).

    We render the scene with the path tracer integrator and compare the recorded  power
    density with our theoretical expectation.
    We expect the average value to be \\pi/4 * L (pi/4 is the integral of cos^2 for the hemisphere),
    with L being the base radiance <radiance>
    """

    scene_dict = {
        'type': 'scene',
        'sensor': make_sensor(origin=[0,0,0], direction=[0,1,0]),
        'emitter': get_envmap(radiance),
        'integrator': {'type': 'path'}
    }

    scene = mi.load_dict(scene_dict)
    scene.integrator().render(scene, seed=0, spp=3200)
    film = scene.sensors()[0].film()

    img = film.bitmap(raw=True).convert(mi.Bitmap.PixelFormat.Y,
                                        mi.Struct.Type.Float32, srgb_gamma=False)
    img_1 = dr.copy(mi.TensorXf(img)).numpy()

    scene_dict = {
        'type': 'scene',
        'sensor': make_sensor(origin=[0,0,0], direction=[0,0,1], radiancemeter=True),
        'emitter': get_envmap(radiance),
        'integrator': {'type': 'path'}
    }
    scene = mi.load_dict(scene_dict)

    scene.integrator().render(scene, seed=0, spp=3200)
    film = scene.sensors()[0].film()

    img = film.bitmap(raw=True).convert(mi.Bitmap.PixelFormat.Y,
                                        mi.Struct.Type.Float32, srgb_gamma=False)

    img_2 = mi.TensorXf(img).numpy()

    assert np.allclose(img_1, img_2), "Measured: " + str(img_1) + " Ref (rad): " + str(img_2 )



def sensor_disk_dict(origin=[0,0,0], target=[0,1,0], radius=1):
    return {
        'type': 'disk',
        'to_world': mi.ScalarTransform4f().look_at(origin, target, [0, 0, 1]).scale(radius),
        'sensor': {
            'type': 'irradiancemeter',
            'film': {
                'type': 'hdrfilm',
                'width': 1,
                'height': 1,
                'rfilter': {'type': 'box'}
            },
        }
    }


@pytest.mark.parametrize(
    ("radiance", "angle"),
    [(2.04, 90), (1.0, 0.0), (1000.0, 30.0)]
)
def test_vs_disk(variants_any_llvm, radiance, angle):
    """
    We test the recorded power density of the vector irradiance meter, by creating a simple scene:
    The irradiance meter is at the coordinate origin looking towards the given direction,
    surrounded by an envmap with a cosine weighted radiance (L(theta)=cos(theta)).

    We render the scene with the path tracer integrator and compare the recorded  power
    density with our theoretical expectation.
    We expect the average value to be \\pi/4 * L (pi/4 is the integral of cos^2 for the hemisphere),
    with L being the base radiance <radiance>
    """
    d = [0,0,1]

    scene_dict = {
        'type': 'scene',
        'sensor': make_sensor(origin=[0,0,0], direction=[0,1,0]),
        'emitter': get_envmap(radiance),
        'integrator': {'type': 'path'}
    }

    scene = mi.load_dict(scene_dict)
    scene.integrator().render(scene, seed=0, spp=3200)
    film = scene.sensors()[0].film()

    img = film.bitmap(raw=True).convert(mi.Bitmap.PixelFormat.Y,
                                        mi.Struct.Type.Float32, srgb_gamma=False)
    img_1 = dr.copy(mi.TensorXf(img)).numpy()

    scene_dict = {
        'type': 'scene',
        'sensor': sensor_disk_dict(origin=[0,0,0], target=d),
        'emitter': get_envmap(radiance),
        'integrator': {'type': 'path'}
    }
    scene = mi.load_dict(scene_dict)

    scene.integrator().render(scene, seed=0, spp=3200)
    film = scene.sensors()[0].film()

    img = film.bitmap(raw=True).convert(mi.Bitmap.PixelFormat.Y,
                                        mi.Struct.Type.Float32, srgb_gamma=False)

    img_2 = mi.TensorXf(img).numpy()

    assert np.allclose(img_1, img_2), "Measured: " + str(img_1) + " Ref (disk): " + str(img_2 )
