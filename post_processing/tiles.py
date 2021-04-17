"""
tiles.py
Zhiang Chen, Jan 7 2020
To process tornado damage tiles
"""

import os
import numpy as np
import pickle
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import matplotlib.pyplot as plt

class Tiles(object):
    def __init__(self, size=(1000, 1000)):
        self.size = size

    def generateTiles(self, path, cls, threshold=0.75):
        self.path = path  # path.split('_')[0]
        pickle_files = [f for f in os.listdir(path) if f.endswith('pickle')]
        assert len(pickle_files)
        for pickle_file in pickle_files:
            image_file = os.path.join(path, pickle_file.split('.')[0] + ".png")
            f = os.path.join(path, pickle_file)
            with open(f, 'rb') as filehandle:
                data = pickle.load(filehandle)
            tile = self.__createTile(data, cls, threshold)
            cv2.imwrite(image_file, tile)

    def __createTile(self, data, cls, threshold):
        boxes = data['boxes']
        labels = data['labels']
        scores = data['scores']
        masks = data['masks']
        image_name = data['image_name']
        image_path = os.path.join('../', image_name)
        assert os.path.isfile(image_path)
        img = cv2.imread(image_path)
        img = img > 10
        img = np.all(img, axis=2)
        h,w = img.shape
        if np.sum(img) < h*w/4.0:
            return np.zeros(self.size).astype(np.uint8)

        if len(boxes) == 0:
            return np.zeros(self.size).astype(np.uint8)
        else:
            idx = scores > threshold
            boxes = boxes[idx]
            if len(boxes) == 0:
                return np.zeros(self.size).astype(np.uint8)
            else:
                labels = labels[idx]
                scores = scores[idx]
                masks = masks[idx]

                idx = labels == cls
                boxes = boxes[idx]
                labels = labels[idx]
                scores = scores[idx]
                masks = masks[idx]
                if len(boxes) == 0:
                    return np.zeros(self.size).astype(np.uint8)
                else:
                    if len(boxes) == 0:
                        return np.zeros(self.size).astype(np.uint8)
                    print(image_path)
                    tile = masks.squeeze(axis=1)
                    tile = tile.max(axis=0)
                    for box in boxes:
                        y1, x1, y2, x2 = box
                        pt1 = (y1, x1)
                        pt2 = (y2, x2)
                        tile = cv2.rectangle(tile, pt1, pt2, color=1, thickness=2)
                    tile = (tile * 255).astype(np.uint8)
                    return tile

    def readTiles(self, path, type="grayscale"):
        tile_files = os.listdir(path)
        self.tiles = {}
        for tile_file in tile_files:
            x, y = tile_file.split('.')[0].split('_')
            #   -----> x
            #   |
            #   |
            # y v
            file_path = os.path.join(path, tile_file)
            if type == "grayscale":
                tile = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            else:
                tile = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            self.tiles[(x, y)] = tile



    def concatenate(self, path, name, step, scale, type="grayscale"):
        """
        :param path:
        :param step: the actual size of tile
        :param scale: the actual tile = scale * tile
        :return:
        """
        tile_files = [i for i in os.listdir(path) if i.endswith('.png')]
        X = []
        Y = []
        for tile_file in tile_files:
            #print(tile_file)
            x, y = tile_file.split('.')[0].split('_')[-2: ]
            X.append(int(x))
            Y.append(int(y))
        width = max(X)/scale + step/scale
        height = max(Y)/scale + step/scale

        if type == "grayscale":
            map = np.zeros((int(height), int(width)))
        else:
            map = np.zeros((int(height), int(width),  3))

        for tile_file in tile_files:
            x, y = tile_file.split('.')[0].split('_')[-2: ]
            x, y = int(int(x)/scale), int(int(y)/scale)
            file_path = os.path.join(path, tile_file)
            if type == "grayscale":
                tile = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            else:
                tile = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if tile.shape[0] != step/scale:
                tile = cv2.resize(tile, (int(step/scale), int(step/scale) ))
            map[y:y+int(step/scale), x:x+int(step/scale)] = tile

        cv2.imwrite(name, map)


    def applyMask(self, map_path, mask_path, color=(1, 1, 0)):
        if not map_path.endswith(".tif"):
            map = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        x, y = map.shape[:2]
        x_, y_ = mask.shape[:2]
        if x < x_:
            mask = cv2.resize(mask, (y, x))
        else:
            map = cv2.resize(map, (y_, x_))

        alpha = 0.5
        mask = mask > 1
        for c in range(3):
            map[:, :, c] = np.where(mask == 1,
                                      map[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      map[:, :, c])

        return map



if __name__ == '__main__':
    t = Tiles()
    t.generateTiles('../103_pred/', 1, threshold=0.8)
    #t.generateTiles('../104_pred/', 2, threshold=0.7)
    t.concatenate('../103_pred/non_damaged/', name="non_damaged_103.png", step=2000, scale=2)
    #t.concatenate('../104_pred/damaged/', name="damaged_104.png", step=2000, scale=2)
    #t.concatenate('../101_pred/damaged_60/', name="damaged_60_101.png", step=2000, scale=2)
    #t.concatenate('../104/', name="104.png", step=2000, scale=4, type="rgb")
    #mask_map = t.applyMask('101.png', 'damaged_101.png', color=(1, 0, 1))
    #cv2.imwrite("masked_damaged_101.png", mask_map)
    mask_map = t.applyMask('masked_damaged_103.png', 'non_damaged_103.png', color=(0, 1, 1))
    cv2.imwrite("masked_103.png", mask_map)
