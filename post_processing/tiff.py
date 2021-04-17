import gdal
import matplotlib.image as mpimg
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 3000000000
import os
import numpy as np

def image2tiff(image_file, tiff_file, save_name):
	assert os.path.isfile(image_file)
	assert os.path.isfile(tiff_file)
	print('reading image')
	img = mpimg.imread(image_file)
	ds = gdal.Open(tiff_file)
	print('reading tiff')
	R = ds.GetRasterBand(1).ReadAsArray()
	#G = ds.GetRasterBand(2).ReadAsArray()
	#B = ds.GetRasterBand(3).ReadAsArray()
	dim = R.shape	
	driver = gdal.GetDriverByName('GTiff')
	dataset = driver.Create(save_name, dim[1], dim[0], 3, gdal.GDT_Byte)
	md = ds.GetMetadata()
	gt = ds.GetGeoTransform()
	pj = ds.GetProjection()
	dataset.SetGeoTransform(gt)
	dataset.SetMetadata(md)
	dataset.SetProjection(pj)
	print('converting')
	dataset.GetRasterBand(1).WriteArray(img[:dim[0], :dim[1], 0])
	dataset.GetRasterBand(2).WriteArray(img[:dim[0], :dim[1], 1])
	dataset.GetRasterBand(3).WriteArray(img[:dim[0], :dim[1], 2])
	print('saving')
	dataset.FlushCache()


image2tiff("heatmap_103_true3.jpg", "eureka_vis103_orth.tif", "heatmap_103_true3.tiff")
image2tiff("heatmap_103_true2.jpg", "eureka_vis103_orth.tif", "heatmap_103_true2.tiff")

image2tiff("heatmap_101_true3.jpg", "eureka_vis101_orth.tif", "heatmap_101_true3.tiff")
image2tiff("heatmap_101_true2.jpg", "eureka_vis101_orth.tif", "heatmap_101_true2.tiff")

#image2tiff("heatmap_101_3.jpg", "eureka_vis101_orth.tif", "heatmap_101_3.tiff")
#image2tiff("heatmap_101_2.jpg", "eureka_vis101_orth.tif", "heatmap_101_2.tiff")

image2tiff("heatmap_104_true3.jpg", "eureka_vis104_orth.tif", "heatmap_104_true3.tiff")
image2tiff("heatmap_103_true2.jpg", "eureka_vis104_orth.tif", "heatmap_104_true2.tiff")

#image2tiff("heatmap_104_3.jpg", "eureka_vis104_orth.tif", "heatmap_104_3.tiff")
#image2tiff("heatmap_103_2.jpg", "eureka_vis104_orth.tif", "heatmap_104_2.tiff")

