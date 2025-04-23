from sionna.rt import load_scene
from sionna.rt.utils import fibonacci_lattice, spawn_ray_from_sources

import mitsuba as mi
import drjit as dr
import numpy as np
import torch
import sys
import time

class GraphMaker:
    def __init__(self, mitsuba_scene, rays_per_source=10000):
        self.scene = load_scene(filename=mitsuba_scene)

        self.mi_scene = self.scene.mi_scene
        self.rays_per_source = rays_per_source
        self.shapes = mi.MeshPtr(self.mi_scene.shapes_dr())

        self.face_normals = self.calc_face_normals()
        self.prim_vertices, self.shape_ptr_to_offset = self.__get_vertices_and_offsets()
        self.face_averages = self.prim_vertices.sum(dim=1) / 3

    def get_vertices(self):
        return self.prim_vertices

    def calc_face_normals(self):
        # Will throw an error if there is a non-mesh shape
        shapes = self.shapes
        shapes = mi.MeshPtr(shapes)

        face_normals = [
            s.face_normal(dr.arange(dr.cuda.UInt, s.face_count())).torch().T for s in shapes
        ]
        face_normals = torch.vstack(face_normals)
        return face_normals
    
    def __get_vertices_and_offsets(self):
        shapes = self.shapes
        shape_ptrs = dr.reinterpret_array(mi.UInt32, shapes).numpy()

        prim_vertices = []

        shape_ptr_to_offset = {}
        offset_counter = 0
        for i, s in enumerate(shapes):
            shape_ptr_to_offset[shape_ptrs[i]] = offset_counter
            offset_counter += s.primitive_count()

            indices = dr.arange(dr.cuda.UInt, s.face_count())
            vertex_indices = dr.ravel(s.face_indices(indices), order='F')
            vertices = s.vertex_position(vertex_indices)
            vertices = vertices.torch().T.reshape(-1, 3, 3)
            prim_vertices.append(vertices)

        prim_vertices = torch.vstack(prim_vertices)
        return prim_vertices, shape_ptr_to_offset
        
    def __rt_graph_interactions(self):
        mi_scene = self.mi_scene
        face_averages = self.face_averages
        face_normals = self.face_normals
        rays_per_source = self.rays_per_source
        
        face_averages = mi.Point3f(face_averages.T)
        face_normals = mi.Vector3f(face_normals.T)

        sources = face_averages + (face_normals * 0.0001)

        sky_test = mi.Ray3f(sources, mi.Vector3f([0,0,1]))
        prelim_si = mi_scene.ray_intersect_preliminary(sky_test)
        exterior_source = ~prelim_si.is_valid()

        rays = spawn_ray_from_sources(fibonacci_lattice, rays_per_source, sources) 

        norms_expanded = dr.repeat(face_normals, rays_per_source)
        exterior_source = dr.repeat(exterior_source, rays_per_source)

        active = exterior_source & (dr.dot(rays.d, norms_expanded) > 0)

        # Old way, not necessary?
        # prelim_si = mi_scene.ray_intersect_preliminary(rays, active=active)
        # full_si = prelim_si.compute_surface_interaction(rays, active=active)

        # Some cursed things happening here
        # dr.set_flag(dr.JitFlag.Debug, True)

        # This breaks sometimes, drjit or cuda bug or something bc there is no reason why this shouldnt work
        full_si = mi_scene.ray_intersect(rays, active=active)

        valid = full_si.is_valid().numpy()
        interaction_normals = full_si.n.numpy().T[valid]

        # make sure the dot product is negative
        # outgoing rays dirs
        # maybe keep this in dr.jit
        ray_dir_np = rays.d.numpy().T[valid]
        ray_dir_dot_interaction_normal_np = np.sum(ray_dir_np * interaction_normals, axis=1)
        valid_hits_exterior = ray_dir_dot_interaction_normal_np < 0

        indices = np.nonzero(valid)[0]
        invalid = indices[~valid_hits_exterior]
        valid[invalid] = False

        indices = np.nonzero(valid)[0]
        interaction_shapes = full_si.shape.numpy()[valid]
        prim_indices = full_si.prim_index.numpy()[valid]
        return indices, interaction_shapes, prim_indices, valid
    
    # Could make this faster using np.unique
    def calc_graph_edges_old(self):
        indices, interaction_shapes, prim_indices, valid = self.__rt_graph_interactions()

        count = 0
        for s in self.shapes:
            count += s.primitive_count()

        sets = [set() for _ in range(count)]

        # Absolutely disgusting
        for i, shape_i, prim_i in zip(indices, interaction_shapes, prim_indices):
            offset = self.shape_ptr_to_offset[shape_i]
            sets[i // self.rays_per_source].add(offset + prim_i)

        # When moving to cpu, torch indexes differently so no transpose
        for i, s in enumerate(self.face_averages.cpu().numpy()):
            print(s, sets[i])

        return sets
    
    def calc_graph_edges(self):
        indices, interaction_shapes, prim_indices, valid = self.__rt_graph_interactions()

        valid_2d = valid.reshape((-1, self.rays_per_source))
        print(valid_2d)
        print(valid_2d.shape)
        
        sum = np.sum(valid_2d, axis=1)
        cum_index = np.cumsum(sum)

        shape_offsets = interaction_shapes
        for key in self.shape_ptr_to_offset.keys():
            shape_offsets[shape_offsets == key] = self.shape_ptr_to_offset[key]

        offsets_plus_prim_i = shape_offsets + prim_indices

        prev = 0
        adjacency_list = []    
        for ci in range(len(cum_index)):
            current = cum_index[ci]
            source_offset_plus_prim_indices = offsets_plus_prim_i[prev:current]
            adjacency_row = np.unique(source_offset_plus_prim_indices)
            adjacency_list.append(adjacency_row)
            prev = current

        return adjacency_list 
        

    def calc_antenna_edges(self, antenna_locations):
        sources = mi.Point3f(antenna_locations.T)
        rays = spawn_ray_from_sources(fibonacci_lattice, self.rays_per_source, sources)
        prelim_int = self.mi_scene.ray_intersect_preliminary(rays)
        valid = prelim_int.is_valid().numpy()
        prim_indices = prelim_int.prim_index.numpy()[valid]
        shape_i = prelim_int.shape.numpy()[valid]

        #reshape valid into (num tx, ray_per_s)
        valid_2d = valid.reshape((antenna_locations.shape[0], -1))
        sum = np.sum(valid_2d, axis=1)
        cum_index = np.cumsum(sum)

        shape_offsets = shape_i
        for key in self.shape_ptr_to_offset.keys():
            shape_offsets[shape_offsets == key] = self.shape_ptr_to_offset[key]

        offsets_plus_prim_i = shape_offsets + prim_indices

        prev = 0
        adjacency_list = []    
        for ci in range(len(cum_index)):
            current = cum_index[ci]
            source_offset_plus_prim_indices = offsets_plus_prim_i[prev:current]
            adjacency_row = np.unique(source_offset_plus_prim_indices)
            adjacency_list.append(adjacency_row)
            prev = current

        #need array of tx specific offsets + 
        #need unique offset + shape_index for each tx
        return adjacency_list 

if __name__ == "__main__":
    gm = GraphMaker("minis2/row4_col4.xml")
    print(gm.calc_graph_edges())

    time.sleep(5)

    pos = np.genfromtxt(fname="output_64maps_10txs_5000rx/row0_col0/tx_locations")
    print(gm.calc_antenna_edges(pos))