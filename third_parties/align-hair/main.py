import trimesh
import cv2 as cv

import os
import numpy as np
import sys

hair_icp_indices = np.array([
    4, # above left ear
    36, # above right ear
    524, # head
    502, # 
    371, # 
    371, # 
    371, # 
])


body_icp_indices = np.array([
    7195, # above left ear
    7990, # above right ear
    10422, # head
    10444, #
    10468, # 
    10468, # 
    10468, # 
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
    d /= np.linalg.norm(d, axis=0)

    o_forward_indices= np.sum(mesh_hair.vertex_normals * d, axis=1) < 0

    mesh_hair.vertices[o_forward_indices] = o

    mesh_merged = trimesh.util.concatenate([mesh_body, mesh_hair])
    
    return mesh_merged

if __name__ == '__main__':
    output_path = sys.argv[1]
    hair_path = sys.argv[2]
    hair_mask_path = sys.argv[3]

    for exp in os.listdir(output_path):
        if not os.path.isdir(os.path.join(output_path, exp)):
            continue

        body_obj_path = os.path.join(output_path, exp, f'{exp}_result.obj')

        save_dir = os.path.join(output_path, exp, 'result_with_hair')
        os.makedirs(save_dir, exist_ok=True)

        mesh_body = trimesh.load_mesh(body_obj_path, process=False)
        mesh_hair = trimesh.load_mesh(hair_path, process=False)

        mesh_merged = align_head_mesh(mesh_body, mesh_hair)

        # mesh_merged.visual.uv[mesh_body.vertices.shape[0]:] = [0.75, 0.5]
        mesh_merged.export(os.path.join(save_dir, f'{exp}_result_with_hair.obj'))
        
        # post-process material
        hair_img = cv.imread(os.path.join(hair_mask_path, f'{exp}_hair_only.png'))
        idx = np.all(hair_img[:,:] != [0, 0, 0], axis=2)
        hair_img_nonzero = hair_img[idx]
        # hair_img_nonzero = hair_img[]
        print(hair_img_nonzero.shape)
        hair_color = hair_img_nonzero.mean(axis=0).astype(np.uint8)

        hair_color_hsv = cv.cvtColor(hair_color.reshape((1,1,3)), cv.COLOR_BGR2HSV)
        # hair_color_hsv[0][0][2] *= 1.5 # 50% brighter
        # hair_color = cv.cvtColor(hair_color_hsv, cv.COLOR_HSV2BGR)
        
        img = cv.imread(os.path.join(save_dir, 'material_0.png'))
        h, w, _ = img.shape
        
        hair_tex = img[256:-256,:] 

        tex_mask = cv.inRange(hair_tex, 0,1)

        tex_mask = 255 - tex_mask

        tex_mask = (tex_mask // 255).astype(bool)

        hsv = cv.cvtColor(hair_tex, cv.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float64)
        v = hsv[tex_mask, 2]
        mean_v = v.mean()

        d_v = (v - mean_v).astype(np.float64) * 0.5

        hsv[tex_mask] = hair_color_hsv
        hsv[tex_mask,2] += d_v

        hsv[hsv[:,:,2] < 0] = 0
        hsv = hsv.astype(np.uint8)

        img[256:-256,:] = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        cv.imwrite(os.path.join(save_dir, 'material_0.png'), img)

        # edit material.mtl
        material_path = os.path.join(save_dir, 'material.mtl')
        with open(material_path,'rt') as f:
            content = f.readlines()
            content[5] = "Ks 1.00000000 1.00000000 1.00000000\n"

        with open(material_path, "wt")  as f:
            f.writelines(content)