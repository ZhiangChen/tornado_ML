"""
data.py
Zhiang Chen, June 2020
Custom dataset for Eureka image classification (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
"""


from __future__ import print_function, division
import os
import json
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

normalize = transforms.Normalize(mean=[0.44, 0.50, 0.43],
                                     std=[0.26, 0.25, 0.26])

data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,])


class EurekaDataset(Dataset):
	def __init__(self, root_dir, json_file, transform=None, binary_cls=False):
		assert os.path.isdir(root_dir)
		assert os.path.isfile(json_file)
		self.root_dir = root_dir
		self.classes = ['non_damage','light_damage','severe_damage']
		self.images = os.listdir(root_dir)
		self.dataset = []
		self.__readJson(json_file)
		self.transform = transform
		self.binary_cls = binary_cls

	def __readJson(self, json_file):
		with open(json_file, 'r') as f:
			labels = json.load(f)	
		for label in labels:
			if label['Label']:
				if label['Label'] == 'Skip':
					continue
				dataset_name = label['Dataset Name']
				image_name = label['External ID']
				image_file = dataset_name + '_' + image_name
				if image_file in self.images:
					#print(label['Label'])
					if 'tile_damage' in label['Label']:
						data = {}
						image_path = os.path.join(self.root_dir, image_file)
						clas = self.classes.index(label['Label']['tile_damage'])
						data['image_path'] = image_path
						data['label'] = clas
						self.dataset.append(data)
						if clas == 2:
							#print(image_file)
							for i in range(1,8):
								data = {}
								image_file = str(i) + '_' + image_file
								if image_file in self.images:
									image_path = os.path.join(self.root_dir, image_file)
									data['image_path'] = image_path
									data['label'] = clas
									self.dataset.append(data)	


	def addJson(self, json_file):
		assert os.path.isfile(json_file)
		with open(json_file, 'r') as f:
			labels = json.load(f)
		for label in labels:
			if label['Label']:
				if label['Label'] == 'Skip':
					continue
				dataset_name = label['Dataset Name']
				image_name = label['External ID']
				image_file = dataset_name + '_' + image_name
				if image_file in self.images:
					#print(label['Label'])
					if 'tile_damage' in label['Label']:
						data = {}
						image_path = os.path.join(self.root_dir, image_file)
						clas = self.classes.index(label['Label']['tile_damage'])
						data['image_path'] = image_path
						data['label'] = clas
						self.dataset.append(data)
						if clas == 2:
							#print(image_file)
							for i in range(1,8):
								data = {}
								image_file = str(i) + '_' + image_file
								if image_file in self.images:
									image_path = os.path.join(self.root_dir, image_file)
									data['image_path'] = image_path
									data['label'] = clas
									self.dataset.append(data)

	def __len__(self):
		return len(self.dataset)


	def __getitem__(self, idx):
		image = io.imread(self.dataset[idx]['image_path'])
		label = self.dataset[idx]['label']
		image = Image.fromarray(image)
		image = image.resize((600,600))
		if self.transform:
			image = self.transform(image)
		if self.binary_cls:
			if label>=2:
				label=1
		return image, label


	def show(self, idx):
		image, label = self.__getitem__(idx)
		print(self.classes[label])
		image.show()
		
	def stats(self):
		images = np.empty((0,3), float)
		indice = list(range(len(self.dataset)))
		import random
		random.shuffle(indice)
		for i in indice[:100]:
			img, _ = self.__getitem__(i)
			img = img.reshape(-1,3)/255.0
			images = np.append(images, img, axis=0)
		return np.mean(images, axis=0).tolist(), np.std(images, axis=0).tolist(), \
               np.max(images, axis=0).tolist(), np.min(images, axis=0).tolist()


if __name__  ==  "__main__":
	ds = EurekaDataset('./datasets/Eureka/images/','./datasets/Eureka/class.json', data_transform)
	
