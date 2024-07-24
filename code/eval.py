import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
from matplotlib import pyplot as plt
from model import *
from SSL import *
from sklearn.model_selection import train_test_split

def main(seed, classifier, nlayers, gamma, data, k, c1, c2, device):
    img_size = 14
    batch_size = 4096
    if k!=1:
        nevals = 10
    else:
        nevals = 1
    for _ in range(nevals):
        transforms = {'train': T.Compose([T.Resize((img_size,img_size)), T.ToTensor()]),
                    'random': T.Compose([T.RandomResizedCrop((img_size,img_size)),
                    T.RandomHorizontalFlip()])}
        # LOAD DATASET
        if data == 'MNIST':
            dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms['train'], download = True)
            evalset = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms['train'], download = True)
        elif data == 'FashionMNIST':
            dataset = torchvision.datasets.FashionMNIST(root = './data', train = True, transform = transforms['train'], download = True)
            evalset = torchvision.datasets.FashionMNIST(root = './data', train = False, transform = transforms['train'], download = True)
        elif data == 'KMNIST':
            dataset = torchvision.datasets.KMNIST(root = './data', train = True, transform = transforms['train'], download = True)
            evalset = torchvision.datasets.KMNIST(root = './data', train = False, transform = transforms['train'], download = True)
        else:
            print('DATASET not implemented!')

        # SELECT CLASS
        class_idx = (dataset.targets == c1) | (dataset.targets == c2)
        dataset.targets = dataset.targets[class_idx]
        dataset.data = dataset.data[class_idx]
        eval_class_idx = (evalset.targets == c1) | (evalset.targets == c2)
        evalset.targets = evalset.targets[eval_class_idx]
        evalset.data = evalset.data[eval_class_idx]
        # SAMPLER
        if k == 1:
            indices = np.arange(len(evalset))
        else:
            indices = np.arange(len(evalset))
            indices, _ = train_test_split(indices, train_size = int(len(evalset)/k), stratify = evalset.targets)
            evalset = torch.utils.data.Subset(evalset, indices)
        # LOAD QUEUE
        queue = torch.utils.data.DataLoader(evalset, batch_size=batch_size, pin_memory = True, shuffle=True, num_workers = 10)
        # LOAD MODEL
        if classifier == 'quantum':
            model = QuantumClassifier(nlayers, device).to(device)
        elif classifier == 'classical':
            model = ClassicalClassifier(nlayers, device).to(device)
        else:
            print('Choose Model: quantum or classical')
        nparams = count_parameters(model)
        nsamples = len(dataset)
        study = 'train__data_{}__c1_{}__c2_{}__nsamples_{}__seed_{}__classifier_{}__nparams_{}__nlayers_{}__gamma_{}'.format(data,c1,c2,nsamples,seed,classifier,nparams,nlayers,gamma)
        model.load_state_dict(torch.load('{}/best_model.pth'.format(study)))
        nsamples_eval = len(evalset)
        print(nsamples_eval)
        out_study = 'eval__data_{}__c1_{}__c2_{}__seed_{}__classifier_{}__nparams_{}__nlayers_{}__gamma_{}'.format(data,c1,c2,seed,classifier,nparams,nlayers,gamma)
        try:
            os.mkdir(out_study)
        except:
            pass
        correct = 0
        model.eval()
        for i, (inputs, targets) in enumerate(queue):
            inputs = inputs.to(device)
            targets = targets.to(device)
            x = model(inputs)
            x = torch.argmax(x, dim = 1)
            correct +=  (x == targets).float().sum()
            print(correct/len(evalset))
        torch.save(model.state_dict(),'{}/best_model__k_{}__nsamples_{}__acc_{:.8f}.pth'.format(out_study, k, len(evalset),correct/len(evalset)))
        acc = correct/len(evalset)
        acc = acc.item()
        if acc < 0.6:
            break

