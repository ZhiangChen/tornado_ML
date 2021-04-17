"""
training module
Zhiang Chen, Oct 2
It includes sampling, training, and inference for training data
"""

from sampling import SP
import cv2
import numpy as np
import torch
import gpytorch
import os
import matplotlib.pyplot as plt
import sys

iteration = int(sys.argv[-1])

print(iteration)
fhandle = open("log/training_log_" + str(iteration) + ".txt", 'w')
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


sp = SP()
detect_file = 'resized_img_detect.png'
clas_file = 'resized_img_clas.png'
mask_file = 'resized_mask.png'
sp.readOriginal(detect_file, clas_file, mask_file)
grid = sp.grid_images(200)
sp.sample()

train_file = 'train_data.npy'
assert os.path.isfile(train_file)
train_data = np.load(train_file)
train_x = np.array(np.nonzero(train_data)).transpose()
train_y = np.array([train_data[tuple(i)] for i in train_x])

# convert to torch tensor
train_x = torch.tensor(train_x, dtype=torch.float)
train_y = torch.tensor(train_y, dtype=torch.float)

# create model and likelihood

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# use GPU
train_x = train_x.cuda()
train_y = train_y.cuda()
model = model.cuda()
likelihood = likelihood.cuda()

# set train model
model.train()
likelihood.train()

# use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 80
log = "Iteration " + str(iteration) + '\n'
fhandle.write(log)
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    log = 'Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ) + '\n'
    fhandle.write(log)
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

model.eval()
likelihood.eval()

# create test data
test_data = cv2.imread('mask_grid.png', cv2.IMREAD_GRAYSCALE)
#test_data = cv2.imread('clas_grid.png', cv2.IMREAD_GRAYSCALE)
test_x = np.array(np.nonzero(test_data)).transpose()
test_x = torch.tensor(test_x, dtype=torch.float)
test_x = test_x.cuda()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()

# inference results
mean = mean.cpu().numpy()
lower = lower.cpu().numpy()
upper = upper.cpu().numpy()

train_x = train_x.cpu().numpy()
train_y = train_y.cpu().numpy()
test_x = test_x.cpu().numpy()

result_grid = np.zeros_like(test_data, dtype=float)
#result_grid = np.zeros_like(test_data)
model_mean = model.state_dict()['mean_module.constant'].cpu().numpy()[0]

fhandle.write("model mean:"+str(model_mean)+'\n')
for i, (u,v) in enumerate(test_x):
    if mean[i] > 5:
        result_grid[int(u), int(v)] = 5
    elif mean[i] < model_mean + 0.05:
        result_grid[int(u), int(v)] = 1
    else:
        result_grid[int(u), int(v)] = mean[i]

#plt.imshow(result_grid)
#plt.show()
#cv2.imwrite('result_grid.png', result_grid)

# variance map
var = np.subtract(upper, lower)
var = var/2
var = np.multiply(var, var)
var_grid = np.zeros_like(test_data, dtype=float)
for i, (u,v) in enumerate(test_x):
    var_grid[int(u), int(v)] = var[i]

#plt.imshow(var_grid)
#plt.show()
#cv2.imwrite('var_grid.png', var_grid)

# precision map
prec_grid = np.zeros_like(test_data, dtype=float)
for i, (u,v) in enumerate(test_x):
    prec_grid[int(u), int(v)] = 1./var[i]

#plt.imshow(prec_grid)
#plt.show()
#cv2.imwrite('prec_grid.png', prec_grid)

# save results
results = np.zeros((200, 200, 3), dtype=float)  # mean, variance, precision
results[:,:,0] = result_grid
results[:,:,1] = var_grid
results[:,:,2] = prec_grid

np.save('results/gp_inference_mean_var_prec_'+ str(iteration) + '.npy', results)
