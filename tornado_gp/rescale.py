"""
resize tif files
Zhiang Chen
Sept 2020
"""

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import gdal
import cv2
import numpy as np
import PIL.Image
import os
PIL.Image.MAX_IMAGE_PIXELS = 3000000000

class RRC(object):
    def __init__(self, tif_detect, tif_clas, xsize_detect=57094, ysize_detect=57234, xsize_clas=71975, ysize_clas=57235):
        self.tif_detect = tif_detect
        self.tif_clas = tif_clas
        self.XSize_detect = xsize_detect
        self.YSize_detect = ysize_detect
        self.XSize_clas = xsize_clas
        self.YSize_clas = ysize_clas

    def readTiff(self, detect_tif, clas_tif):
        self.detect_ds = gdal.Open(detect_tif)
        self.XSize_detect = self.detect_ds.RasterXSize
        self.YSize_detect = self.detect_ds.RasterYSize
        band_detect = self.detect_ds.GetRasterBand(1)
        self.clas_ds = gdal.Open(clas_tif)
        self.XSize_clas = self.clas_ds.RasterXSize
        self.YSize_clas = self.clas_ds.RasterYSize
        band_clas = self.clas_ds.GetRasterBand(1)
        return band_detect, band_clas

    def rotate(self, image, angle):
        image_center = tuple(np.array(image.shape[:2]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
        return result

    def get_resized_mask(self):
        band_detect, band_clas = self.readTiff(self.tif_detect, self.tif_clas)
        tif_detect = band_detect.ReadAsArray()
        tif_clas = band_clas.ReadAsArray()
        print(tif_detect.shape)
        print(tif_clas.shape)
        resize_tif_clas = cv2.resize(tif_clas, None, fx=0.1, fy=0.1)
        U,V = resize_tif_clas.shape
        resize_tif_detect = cv2.resize(tif_detect, (V, U))
        print(resize_tif_detect.shape)
        print(resize_tif_clas.shape)
        """
        enlarged_resized_tif_detect = np.zeros((U+1000, V+1000))
        enlarged_resized_tif_detect[0:-1000, 0:-1000] = resize_tif_detect
        rot_tif_detect = self.rotate(enlarged_resized_tif_detect, -39)
        cv2.imwrite("resized_rot.png", rot_tif_detect)

        X = np.max(rot_tif_detect, axis=0)
        Y = np.max(rot_tif_detect, axis=1)
        x_nm = X.shape[0]
        y_nm = Y.shape[0]
        for i in range(x_nm):
            if X[i] > 0:
                x_min = i
                break

        for i in range(x_nm - 1, 0, -1):
            if X[i] > 0:
                x_max = i
                break

        for i in range(y_nm):
            if Y[i] > 0:
                y_min = i
                break

        for i in range(y_nm - 1, 0, -1):
            if Y[i] > 0:
                y_max = i
                break

        #print(y_min,y_max, x_min, x_max)
        mask = np.zeros_like(rot_tif_detect)
        #print(mask.shape)
        mask[y_min - 2:y_max + 2, x_min - 2:x_max + 2] = 255
        mask = self.rotate(mask, 39)
        """
        return resize_tif_detect, resize_tif_clas

    def read_sample_npy(self, sample_file):
        assert os.path.isfile(sample_file)
        samples = np.load(sample_file)
        return samples


    def recover_img_size_clas_format(self, image_nd):
        im = PIL.Image.fromarray(image_nd)
        recoverd_im = im.resize((self.XSize_clas, self.YSize_clas), resample=PIL.Image.BICUBIC)
        nd_img = np.array(recoverd_im)
        nd_img[nd_img<0] = 0
        return nd_img


    def save_tiff_as_class_format(self, ndarray, save_file):
        driver = gdal.GetDriverByName('GTiff')
        y, x = ndarray.shape

        dataset = driver.Create(
            save_file,
            x,
            y,
            1,
            gdal.GDT_Byte)

        md = self.clas_ds.GetMetadata()
        gt = self.clas_ds.GetGeoTransform()
        pj = self.clas_ds.GetProjection()
        dataset.SetGeoTransform(gt)
        dataset.SetMetadata(md)
        dataset.SetProjection(pj)
        dataset.GetRasterBand(1).WriteArray(ndarray.astype(np.uint8))
        dataset.FlushCache()




if __name__ == '__main__':
    """
    tif_detect = 'heatmap_detect_102_103_mult_aug.tif'
    tif_clas = 'heatmap_clas_102_103.tif'
    rrc = RRC(tif_detect, tif_clas)
    rrc.readTiff(tif_detect, tif_clas)
    #resized_tif_detect, resized_tif_clas = rrc.get_resized_mask()
    #cv2.imwrite('resized_img_detect.png', resized_tif_detect)
    #cv2.imwrite('resized_img_clas.png', resized_tif_clas)
    results = rrc.read_sample_npy('resized_result_mean_var_prec.npy')
    mean = results[:, :, 0]
    var = results[:, :, 1]
    prec = results[:, :, 2]
    resized_mean = rrc.recover_img_size_clas_format(mean)
    rrc.save_tiff_as_class_format(resized_mean, 'mean.tif')
    """
    tif_detect = 'heatmap_detect_102_103_mult_true.tif'
    tif_clas = 'heatmap_clas_102_103_true.tif'
    rrc = RRC(tif_detect, tif_clas)
    rrc.readTiff(tif_detect, tif_clas)
    #resized_tif_detect, resized_tif_clas = rrc.get_resized_mask()
    #cv2.imwrite('resized_img_detect_true.png', resized_tif_detect)
    #cv2.imwrite('resized_img_clas_trueDe.png', resized_tif_clas)
    mean = rrc.read_sample_npy('resized_result_mean_training.npy')
    # mean = results[:, :, 0]
    # var = results[:, :, 1]
    # prec = results[:, :, 2]
    resized_mean = rrc.recover_img_size_clas_format(mean)
    rrc.save_tiff_as_class_format(resized_mean, 'training_mean.tif')
    mean = rrc.read_sample_npy('resized_result_mean_groundtruth.npy')
    resized_mean = rrc.recover_img_size_clas_format(mean)
    rrc.save_tiff_as_class_format(resized_mean, 'groundtruth_mean.tif')


