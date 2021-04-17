from data import *
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

ds = EurekaDataset('./datasets/Eureka/images_test/','./datasets/Eureka/clas.json')

for data in ds.dataset:
	image_path = data['image_path']
	if data['label'] == 2:
		print(image_path)
		image0 = cv2.imread(image_path)		
		image1 = cv2.flip(image0, 0)
		image2 = cv2.flip(image0, 1)
		image3 = cv2.flip(image1, 1)
		image4 = cv2.rotate(image0, cv2.ROTATE_90_CLOCKWISE)
		image5 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
		image6 = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)
		image7 = cv2.rotate(image3, cv2.ROTATE_90_CLOCKWISE)
		image_name = image_path.split('/')[-1]
		path = image_path.split('Eureka_')[0]
		cv2.imwrite(path+'1_'+image_name, image1)
		cv2.imwrite(path+'2_'+image_name, image2)
		cv2.imwrite(path+'3_'+image_name, image3)
		cv2.imwrite(path+'4_'+image_name, image4)
		cv2.imwrite(path+'5_'+image_name, image5)
		cv2.imwrite(path+'6_'+image_name, image6)
		cv2.imwrite(path+'7_'+image_name, image7)		

	if data['label'] == 1:
		image = cv2.imread(image_path)
