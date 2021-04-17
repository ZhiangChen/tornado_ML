"""
training.py
Zhiang Chen, April 2020
"""

import torch
import torch.utils.data
import torchvision.datasets
import torch.nn as nn
import torchvision.transforms as transforms
from utils import *
import torchvision.models as models
from data import EurekaDataset
import pickle

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

torch.manual_seed(0)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


eureka_normalize = transforms.Normalize(mean=[0.44, 0.50, 0.43],
                                     std=[0.26, 0.25, 0.26])

eureka_transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    eureka_normalize,])


def neural_network(architecture, nm_classes, pretrained=True, change_last_layer=True):
    assert architecture in model_names
    print("=> creating model '{}'".format(architecture))
    model = models.__dict__[architecture](pretrained=pretrained)
    if change_last_layer:
        if architecture.startswith('densenet'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features=in_features, out_features=nm_classes)
        else:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features=in_features, out_features=nm_classes)

    return model

def cifar10(root='./datasets/cifar10/', val=True):
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=test_transform)
    return train, test

def eureka():
	train = EurekaDataset('./datasets/Eureka/images/','./datasets/Eureka/class.json', eureka_transform, True)
	#test = EurekaDataset('./datasets/Eureka/images_valid/','./datasets/Eureka/class.json', eureka_transform, True)
	#test.addJson('./datasets/Eureka/label_102.json')
	test = EurekaDataset('./datasets/Eureka/images_test/','./datasets/Eureka/label_102.json', eureka_transform, True)
	return train, test

if __name__ == '__main__':
    cuda = 'cuda:0'
    device = torch.device(cuda)
    nm_classes = 2
    train_dataset, test_dataset = eureka()

    print(len(train_dataset))
    print(len(test_dataset))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=8, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=8, collate_fn=collate_fn)

    model = neural_network('densenet161', nm_classes)

    criterion = nn.CrossEntropyLoss().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.65)

    """
    model.load_state_dict(torch.load("trained_param2_densenet161/epoch_{:04d}.param".format(56)))
    model.eval()    
    model.to(device)
    results = infer(model, train_dataset, device)
    meta = train_dataset.dataset
    save_f = {"pred":results, "meta": meta}
    with open('2_101104_result.pickle', 'wb') as fp:
        pickle.dump(save_f, fp)
    """
    model.load_state_dict(torch.load("trained_param2_densenet161/epoch_{:04d}.param".format(56)))
    model.eval()
    model.to(device)
    results = infer(model, test_dataset, device)
    meta = test_dataset.dataset
    save_f = {"pred":results, "meta": meta}
    with open('2_102_result.pickle', 'wb') as fp:
        pickle.dump(save_f, fp)

