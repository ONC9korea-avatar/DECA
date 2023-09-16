import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans

import sys
import os

import queue

def extract_face_rgb(skin_img_path):
    skin_img = cv2.imread(skin_img_path)

    skin_hsv = cv2.cvtColor(skin_img, cv2.COLOR_BGR2HSV)

    skin_segment = skin_hsv[:, :, 2] > 0

    skin_hsv = skin_hsv[skin_segment]
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(skin_hsv)

    bright_idx = np.argmax(kmeans.cluster_centers_[:, 2])
    bright_color = kmeans.cluster_centers_[bright_idx]

    face_rgb = cv2.cvtColor(bright_color.reshape((1,1,3)).astype(np.uint8),cv2.COLOR_HSV2BGR)

    return face_rgb

def main(texture_path,skin_only_path, save_path):
    img = cv2.imread(texture_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    thresh = 255 - thresh
    darkness = img.sum(axis=2) < 255 * 1.1
    thresh[darkness] = 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    sorted_idx = sorted(range(len(contours)), key=lambda x: cv2.contourArea(contours[x]), reverse=True)

    segment = np.zeros_like(img)

    face_idx = sorted_idx[0]
    cv2.drawContours(segment, contours, face_idx,color=(255,255,255),thickness=cv2.FILLED)

    kernel = np.ones((3,3),np.uint8)
    segment = cv2.erode(segment, kernel, iterations=5)

    for i in sorted_idx[1:3]: # 1,2 -> eye segment
        cv2.drawContours(segment, contours, i,color=(255,255,255),thickness=cv2.FILLED)

    segment = segment.astype(bool)

    g_img = np.zeros_like(img)

    g_img[segment] = img[segment]

    texture_region = segment.any(axis=2)

    contours, hier = cv2.findContours(texture_region.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    largest_idx = max(range(len(contours)), key=lambda x: cv2.contourArea(contours[x]))

    begin_con = contours[largest_idx]

    q = queue.Queue()

    visited = texture_region.copy()

    w, h = visited.shape

    for p in begin_con:
        tuple_p = (p[0,1],p[0,0])
        q.put(tuple_p)

    def get_adj_pixel(pos):
        dpos = [(pos[0] + dx, pos[1] + dy) for dx, dy in ((0,1), (0,-1), (1,0), (-1,0), (1,1), (1, -1), (-1,1), (-1, -1))]
        return filter(lambda x: 0 <= x[0] < w and 0 <= x[1] < h, dpos)

    face_rgb = extract_face_rgb(skin_only_path)

    colors = []

    while not q.empty():
        count = len(q.queue)

        if len(colors) > 0:
            mean_color = np.mean(colors, axis=0)
        else:
            mean_color = None

        colors = []
        for i in range(count):
            p = q.get()

            color = []
            for adj_p in get_adj_pixel(p):
                if not visited[adj_p] and adj_p not in q.queue:
                    q.put(adj_p)
                
                if visited[adj_p]:
                    color.append(g_img[adj_p])

            if mean_color is not None:
                g_img[p] = np.mean(color, axis=0) * 0.95 + mean_color * 0.02 + face_rgb * 0.03
            else:
                g_img[p] = np.mean(color, axis=0)
            
            colors.append(g_img[p])

            visited[p] = True

    cv2.imwrite(save_path,g_img)

if __name__ == '__main__':
    deca_out_path = sys.argv[1]
    save_dir_path = sys.argv[2]
    skin_only_dir_path = sys.argv[3]

    for exp in os.listdir(deca_out_path):
        human_name = '_'.join(exp.split('_')[:-1])

        texture_dir_path = os.path.join(deca_out_path,exp)

        texture_name = os.path.split(texture_dir_path)[-1]
        texture_img_path = os.path.join(texture_dir_path, f'{texture_name}.png')

        skin_only_path = os.path.join(skin_only_dir_path, f'{human_name}_skin_only.png')
        save_path = os.path.join(save_dir_path, f'{texture_name}_filled.png')

        main(texture_img_path, skin_only_path ,texture_img_path)
    
