import numpy as np
import tensorflow as tf
import os

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, ITURadioMaterial, PathSolver

import mitsuba as mi
import drjit as dr


def log_flat_arr(output_dir, file, X):
    shape = X.shape
    flattened = X.reshape(shape[0], -1)
    np.savetxt(fname=os.path.join(output_dir, file),
               X=flattened, header=f'{shape}')


def trace_scene(scene_dir, scene_file):
    scene = load_scene(os.path.join(scene_dir, scene_file))
    output_dir = os.path.join('output', os.path.splitext(scene_file)[0])

    try:
        os.mkdir(output_dir)
    except OSError:
        print('dir already exists')

    rx_points = []

    x_start, y_start, _ = min
    x_end, y_end, _ = max
    xx = np.linspace(x_start, x_end, h_spacing)
    yy = np.linspace(y_start, y_end, h_spacing)
    ray_pos = np.array([(x, y, 1000) for x in xx for y in yy])

    z_rays = mi.Ray3f(o=ray_pos.T, d=(0, 0, -1))
    z_hits = scene.mi_scene.ray_intersect(z_rays)

    for z_hit, valid in zip(z_hits.p.numpy().T, z_hits.is_valid()):
        if (not valid):
            continue
        start = int(z_hit[2] + 10)

        for i in range(start, 100, v_spacing):
            rx_points.append(np.array([z_hit[0], z_hit[1], i]))

    # Include points that did not have a valid hitpoint due to lack of floor right now
    for i, v in enumerate(z_hits.is_valid()):
        if (not v):
            r = ray_pos[i]
            for i in range(5, 100, v_spacing):
                rx_points.append(np.array([r[0], r[1], i]))

    scene.frequency  # Defaults to 3.5GHz

    mat = ITURadioMaterial(name='concrete', itu_type='concrete',
                           thickness=0.1, scattering_coefficient=0.0, xpd_coefficient=0.0)

    # Set all objects to concrete for now
    for obj in scene.objects.values():
        obj.radio_material = mat

    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=1,
                                 horizontal_spacing=1,
                                 polarization="V",
                                 pattern="iso")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 polarization="V",
                                 pattern="iso")

    # Create receivers
    for rx_i in range(len(rx_points)):
        rx = Receiver(name=f"rx_{rx_i}",
                      position=rx_points[rx_i],
                      orientation=[0, 0, 0])
        scene.add(rx)

    points = np.array(rx_points)
    filtered = points[(points[:, 2] <= 50)]
    filtered[:, 2] += 5

    np.random.seed(42)
    c = np.random.choice(filtered.shape[0], size=num_tx, replace=False)
    print(c)

    tx_pos = filtered[c]

    for tx_i in range(num_tx):
        tx = Transmitter(name=f"tx_{tx_i}",
                         position=tx_pos[tx_i],
                         orientation=[0, 0, 0])
        scene.add(tx)

    p_solver = PathSolver()
    paths = p_solver(scene=scene,
                     max_depth=6,
                     samples_per_src=1000000,
                     los=False,
                     specular_reflection=True,
                     diffuse_reflection=False,
                     refraction=False,
                     synthetic_array=False,
                     seed=41)

    tx_log = paths.sources.numpy().T
    log_flat_arr(output_dir, "tx_locations", tx_log)

    rx_log = paths.targets.numpy().T
    log_flat_arr(output_dir, "rx_locations", rx_log)

    real_a_log = paths.a[0].numpy()
    imaginary_a_log = paths.a[1].numpy()
    log_flat_arr(output_dir, "real_a", real_a_log)
    log_flat_arr(output_dir, "imaginary_a", imaginary_a_log)

    valid_log = paths.valid.numpy()
    log_flat_arr(output_dir, "valid", valid_log)

    primitives_log = paths.primitives.numpy()
    log_flat_arr(output_dir, "primitives", primitives_log)

    vertices_log = paths.primitives.numpy()
    log_flat_arr(output_dir, "vertices", vertices_log)

    del scene


if __name__ == "__main__":
    min = (0, 0, 0)
    max = (275, 275, 500)

    h_spacing = 25
    v_spacing = 25
    num_tx = 10

    scene_dir = "minis3"
    scene_files = [f for f in os.listdir(scene_dir) if f.endswith('.xml')]
    print(scene_files)

    for scene_file in scene_files:
        try:
            trace_scene(scene_dir, scene_file)
        except:
            print(f'OOMED {scene_file}')
            continue
