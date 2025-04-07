import bpy
import numpy as np
from mathutils import Vector

export_path = "/home/tyler/Documents/research/dataset/minis/"
num_x_bins = 8
num_y_bins = 8

# The .blend file needs to have buildings as separate objects and without ground plane
# Get all object locations

# It might be better to use bounding boxes, 
# but if bins are large enough then it shouldnt matter
locations = [obj.location for obj in bpy.data.objects]
locations = np.array(locations)

x_min = locations[:, 0].min()
x_max = locations[:, 0].max() + 0.01
x_step = (x_max - x_min) / num_x_bins

y_min = locations[:, 1].min()
y_max = locations[:, 1].max() + 0.01
y_step = (y_max - y_min) / num_y_bins

# Create object bins aligned with xy axis
# If scene is not axis aligned box, probably do that
grid = np.empty((num_x_bins, num_y_bins), dtype=object)
for i in range(num_x_bins):
    for j in range(num_y_bins):
        grid[i][j] = []

for obj in bpy.data.objects:
    x_bin = int((obj.location[0] - x_min) // x_step)
    y_bin = int((obj.location[1] - y_min) // y_step)
    grid[x_bin][y_bin].append(obj)

# Export each grid bin to a separate file
for i in range(num_x_bins):
    for j in range(num_y_bins):
        bpy.ops.object.select_all(action='DESELECT')

        for obj in grid[i][j]:
            # FILTER BASED ON OBJECT COMPLEXITY
            if obj.type == 'MESH':# and len(obj.data.polygons) < 10:
                obj.select_set(True)

        x_mid = num_x_bins / 2
        y_mid = num_y_bins / 2
        x_delta = (x_mid - i) * x_step
        y_delta = (y_mid - j) * y_step
        print(x_delta)
        print(y_delta)

        for obj in bpy.context.selected_objects:
            print(obj.location)
            obj.location.x += x_delta
            obj.location.y += y_delta
            print(obj.location)


        bpy.ops.export_scene.mitsuba(filepath=export_path+f"row{i}_col{j}.xml", export_ids=True, use_selection=True, axis_forward='Y', axis_up='Z')

        for obj in bpy.context.selected_objects:
            obj.location.x -= x_delta
            obj.location.y -= y_delta 
