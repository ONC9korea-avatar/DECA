import trimesh
import numpy as np
import cv2 as cv

import os, sys

def match_head_top(mesh_body, mesh_head):
    body_top = mesh_body.vertices[np.argmax(mesh_body.vertices[:,1])]
    head_top = mesh_head.vertices[np.argmax(mesh_head.vertices[:,1])]

    # print(body_top, head_top)
    T = trimesh.transformations.translation_matrix(body_top-head_top) # initial transformation
    U, *_ = trimesh.registration.icp(mesh_head.vertices, mesh_body, initial=T, scale=True)
    mesh_head.apply_transform(U)

    return mesh_body, mesh_head

def slice_body_and_head(mesh_body, mesh_head):
    o = mesh_body.vertices[3049, :]
    # print(o)
    n = np.array([0., 1., 0.])
    n /= np.linalg.norm(n) # unit normal

    body_cut_mask = mesh_body.vertices[:,1] >= o[1]
    head_cut_mask = mesh_head.vertices[:,1] <= o[1]
    mesh_body.vertices[body_cut_mask] = o[:]
    mesh_head.vertices[head_cut_mask] = o[:]

    # mesh_body = mesh_body.process(validate=True)
    # mesh_head = mesh_head.process(validate=True)

    # mesh_body = mesh_body.slice_plane(o, -n)
    # mesh_head = mesh_head.slice_plane(o, n)

    # return mesh_body, mesh_head

    # ---------- simple align ---------- #
    head_cut_indices = np.where(np.abs(mesh_head.vertices[:,1]-o[1])<0.01)[0]
    body_cut_indices = np.where(np.abs(mesh_body.vertices[:,1]-o[1])<0.01)[0]

    head_cut_vertices = mesh_head.vertices[head_cut_indices, :]
    body_cut_vertices = mesh_body.vertices[body_cut_indices, :]

    # T, *_ = trimesh.registration.icp(head_cut_vertices, body_cut_vertices, scale=False, reflection=False)
    # mesh_head.apply_transform(T)
    # ---------- simple align ---------- #

    # ---------- advanced align ---------- #
    # head_cut_vertices = mesh_head.vertices[head_cut_indices, :]
    # body_cut_vertices = mesh_body.vertices[body_cut_indices, :]

    head_cut_target = []
    for v in head_cut_vertices:
        u = body_cut_vertices[np.argmin(np.linalg.norm(v-body_cut_vertices, axis=1))]
        head_cut_target.append((u+v)/2)
    
    body_cut_target = []
    for v in body_cut_vertices:
        u = head_cut_vertices[np.argmin(np.linalg.norm(v-head_cut_vertices, axis=1))]
        body_cut_target.append((u+v)/2)
    
    mesh_head.vertices[head_cut_indices, :] = head_cut_target
    mesh_body.vertices[body_cut_indices, :] = body_cut_target
    # ---------- advanced align ---------- #

    # ---------- applying skinning weight ---------- #
    # TODO
    # ---------- applying skinning weight ---------- #
    

    return mesh_body, mesh_head


def main(path_body, path_head, save_path):
    for body_obj_name in os.listdir(path_body):
        subject = body_obj_name.split('_')[0]

        f_body = open(os.path.join(path_body, body_obj_name))
        f_head = open(os.path.join(path_head, subject+'_masked', subject+'_masked.obj'))

        mesh_body = trimesh.load_mesh(f_body, 'obj', process=False)
        mesh_head = trimesh.load_mesh(f_head, 'obj', process=False)

        f_body.close()
        f_head.close()

        # print(mesh_body, mesh_head)
        mesh_body, mesh_head = match_head_top(mesh_body, mesh_head)
        mesh_body, mesh_head = slice_body_and_head(mesh_body, mesh_head)


        mesh_merged = trimesh.util.concatenate([mesh_body, mesh_head])
        mesh_merged.visual.uv[:mesh_body.vertices.shape[0]] = [0.75, 0.5]
        # mesh_merged = trimesh.boolean.union([mesh_body, mesh_head], 'blender')
        # # remove_open_faces(mesh_merged)
        # # trimesh.repair.fill_holes(mesh_merged)
        
        os.makedirs(os.path.join(save_path, subject), exist_ok=True)
        obj_path = os.path.join(save_path, subject, subject+'_result.obj')
        mesh_merged.export(obj_path)


        # post-process material
        img = cv.imread(os.path.join(save_path, subject, 'material_0.png'))
        h, w, _ = img.shape

        body_color = img[h-1, w//4, :]
        img[:, w//2:] = body_color

        cv.imwrite(os.path.join(save_path, subject, 'material_0.png'), img)

        # with open(obj_path) as f:
        #     data = f.read().splitlines()
    
        # obj = {}

        # for line in data:
        #     t, *x = line.split()
        #     if t == '#':
        #         continue

        #     if t in obj:
        #         obj[t].append(x)
        #     else:
        #         obj[t] = [x, ]
        
        # # print(obj.keys())
        # # print(len(obj['v']), len(obj['vt']), len(obj['f']))
        # # print(len(mesh_body.vertices), len(mesh_head.vertices))

        # for i in range(mesh_body.vertics.shape[0]):
        #     obj['vt'][i] = 

if __name__ == '__main__':
    path_body, path_head, save_path = sys.argv[1:]
    main(path_body, path_head, save_path)