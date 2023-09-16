import trimesh

import os
import numpy as np
import sys

hair_icp_indices = np.array([
    4, # above left ear
    36, # above right ear
    524, # head
    502, # 
    371 # 
])


body_icp_indices = np.array([
    7195, # above left ear
    7990, # above right ear
    10422, # head
    10444, #
    10468 # 
])

def align_head_mesh(mesh_body, mesh_hair):

    body_head_top = mesh_body.vertices[np.argmax(mesh_body.vertices[:, 1])].reshape(1, 3)
    body_icp_vertices = mesh_body.vertices[body_icp_indices,:]

    hair_head_top = mesh_hair.vertices[np.argmax(mesh_hair.vertices[:, 1])].reshape(1, 3)
    hair_icp_vertices = mesh_hair.vertices[hair_icp_indices,:]

    T = trimesh.transformations.translation_matrix((body_head_top - hair_head_top).reshape(-1))

    T, *_ = trimesh.registration.icp(hair_icp_vertices, body_icp_vertices, scale=True, initial=T, reflection=False)
    mesh_hair.apply_transform(T)

    o = mesh_hair.vertices.mean(axis=0)
    p = mesh_hair.vertices
    d = p - o

    o_forward_indices= np.sum(mesh_hair.vertex_normals * (d / np.linalg.norm(d, axis=0)), axis=1) < 0

    mesh_hair.vertices[o_forward_indices] = o

    mesh_merged = trimesh.util.concatenate([mesh_body, mesh_hair])
    
    return mesh_merged

if __name__ == '__main__':
    output_path = sys.argv[1]
    hair_path = '/workspace/DECA/third_parties/align-hair/hair_mm.obj'

    mesh_hair = trimesh.load_mesh(hair_path, process=False)
    for exp in os.listdir(output_path):
        if not os.path.isdir(os.path.join(output_path, exp)):
            continue

        body_obj_path = os.path.join(output_path, exp,f'{exp}_result.obj')

        save_dir = os.path.join(output_path, exp, 'result_with_hair')
        os.makedirs(save_dir, exist_ok=True)

        mesh_body = trimesh.load_mesh(body_obj_path, process=False)

        mesh_merged = align_head_mesh(mesh_body, mesh_hair)

        mesh_merged.export(os.path.join(save_dir,f'{exp}_result_with_hair.obj'))