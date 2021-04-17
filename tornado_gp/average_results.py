"""
average results
Zhiang Chen, Oct 2
"""

import os
import numpy as np
import matplotlib.pyplot as plt

training = True
if training:
    npy_files = [f for f in os.listdir('results') if 'true' not in f]
else:
    npy_files = [f for f in os.listdir('results') if 'true' in f]


results_all = np.zeros((200, 200, len(npy_files)), dtype=float)
for i, f in enumerate(npy_files):
    result = np.load('results/'+f)
    results_all[:, :, i] = result[:, :, 0]

result_mean = np.mean(results_all, axis=2)
plt.imshow(result_mean)
plt.show()