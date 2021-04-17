import pickle
import os
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

files = [f for f in os.listdir("../datasets/Eureka/images/") if f.startswith("Eureka_104")]
x = []
y = []
for f in files:
	a = f.split('.')[0].split('_')[-2:]
	x.append(int(a[0]))
	y.append(int(a[1]))

x = np.asarray(x)
y = np.asarray(y)

X = x.max()
Y = y.max()

mp = np.zeros((Y+2000, X+2000, 3))
print("map size is (%d, %d)" % (mp.shape[0], mp.shape[1]))

with (open("../datasets/Eureka/2_101104_result.pickle", "rb")) as openfile:
	result = pickle.load(openfile)

pred = result['pred']
meta = result['meta']

for data in zip(pred, meta):
	prd = data[0]
	img = data[1]['image_path']
	lab = data[1]['label']
	name = img.split('/')[-1]
	if name.startswith('Eureka_104'):
		a = img.split('/')[-1].split('.')[0].split('_')[-2:]
		x, y = int(a[0]), int(a[1])
		mx = cv2.resize(cv2.imread('.'+img), (2000, 2000))
		#mp[y:y+2000, x:x+2000, :] = mx
		heatmap = np.zeros((2000, 2000, 3))
		if lab == 0:
			heatmap[:,:,:2] = 255
		elif lab >= 1:
			heatmap[:,:,2] = 255
			heatmap[:,:,1] = 150
		elif lab > 2:
			heatmap[:,:,2] = 255

		mp[y:y+2000, x:x+2000, :] = heatmap

	
cv2.imwrite('heatmap_104_true2.jpg', mp)
	
