from sionna.rt import load_scene, PlanarArray, Receiver, Transmitter, ITURadioMaterial, PathSolver
import numpy as np
import mitsuba as mi

scene = load_scene(filename="minis/row1_col4.xml")

scene.frequency # Defaults to 3.5GHz

mat = ITURadioMaterial(name='concrete', itu_type='concrete', thickness=0.1, scattering_coefficient=0.0, xpd_coefficient=0.0)

# Set all objects to concrete for now
# for obj in scene.objects.values():
#     obj.radio_material = mat

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

tx = Transmitter(name="tx",
              position=[160,100,40],
              orientation=[0,0,0])
scene.add(tx)

# # Create transmitter
# rx = Receiver(name="rx",
#               position=[160,300,40],
#               orientation=[0,0,0])
# scene.add(rx)

rx_points = []
spacing = 50
min = (0, 0, 0)
max = (500, 500, 1000)

x_start, y_start, _ = min
x_end, y_end, _ = max
xx = np.linspace(x_start, x_end, spacing)
yy = np.linspace(y_start, y_end, spacing)
ray_pos = np.array([(x, y, 1000) for x in xx for y in yy])

z_rays = mi.Ray3f(o=ray_pos.T, d=(0, 0, -1))
z_hits = scene.mi_scene.ray_intersect(z_rays)

for z_hit, valid in zip(z_hits.p.numpy().T, z_hits.is_valid()):
    if (not valid):
        continue
    start = int(z_hit[2] + 10)

    for i in range(start, 200, spacing):
        rx_points.append(np.array([z_hit[0], z_hit[1], i]))

# Include points that did not have a valid hitpoint due to lack of floor right now
for i, v in enumerate(z_hits.is_valid()):
    if (not v):
        r = ray_pos[i]
        for i in range(0, 200, spacing):
            rx_points.append(np.array([r[0], r[1], i]))

# Create receivers
for rx_i in range(len(rx_points)):
#     print(rx_points[rx_i])
    rx = Receiver(name=f"rx_{rx_i}",
            position=rx_points[rx_i],
            orientation=[0,0,0])
    scene.add(rx)

p_solver  = PathSolver()
paths = p_solver(scene=scene,
                max_depth=7,
                los=False,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=False,
                synthetic_array=False,
                seed=41)