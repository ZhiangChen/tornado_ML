from skgstat import Variogram
import numpy as np
train_data = np.load('train_data.npy')
train_x = np.array(np.nonzero(train_data)).transpose()
train_y = np.array([train_data[tuple(i)] for i in train_x])
V = Variogram(train_x, train_y)
V.distance_difference_plot()