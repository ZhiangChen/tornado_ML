import os
import numpy as np

def lookup(file_path):
	assert os.path.isfile(file_path)
	f = open(file_path, 'r')
	text = f.read()
	text_ls = text.split('\n')
	acc_ls = []
	for text in text_ls:
		if len(text) >= 20:
			continue
		if len(text) == 0:
			continue
		acc_ls.append(text)
	acc_ls = [float(f[8:]) for f in acc_ls]
	print(acc_ls)
	print(np.max(acc_ls))
	print(np.argmax(acc_ls))

file_path = "trained_param_resnet152/log_102103"
print(file_path)
lookup(file_path)

file_path = "trained_param_wide_resnet/log_102103"
print(file_path)
lookup(file_path)

file_path = "trained_param_resnext101/log_102103"
print(file_path)
lookup(file_path)

file_path = "trained_param_densenet161/log_102103"
print(file_path)
lookup(file_path)

file_path = "trained_param_densenet201/log_102103"
print(file_path)
lookup(file_path)

file_path = "trained_param2_resnet152/log_102103"
print(file_path)
lookup(file_path)

file_path = "trained_param2_wide_resnet/log_102103"
print(file_path)
lookup(file_path)

file_path = "trained_param2_resnext101/log_102103"
print(file_path)
lookup(file_path)

file_path = "trained_param2_densenet161/log_102103"
print(file_path)
lookup(file_path)

file_path = "trained_param2_densenet201/log_102103"
print(file_path)
lookup(file_path)
