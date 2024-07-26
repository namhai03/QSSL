import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
from matplotlib import pyplot as plt
from model import *
from SSL import *


def main(seed, classifier, nlayers, gamma, data, c1, c2, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_epochs = 50
    img_size = 14
    batch_size = 512
    lr =1e-3
    transforms = {'train': T.Compose([T.Resize((img_size,img_size)), T.ToTensor()]),
                'random': T.Compose([T.RandomResizedCrop((img_size,img_size)),
                T.RandomHorizontalFlip()])}
    # LOAD DATASET
    if data == 'MNIST':
        dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms['train'], download = True)
    elif data == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(root = './data', train = True, transform = transforms['train'], download = True)
    elif data == 'KMNIST':
        dataset = torchvision.datasets.KMNIST(root = './data', train = True, transform = transforms['train'], download = True)
    else:
        print('DATASET not implemented!')

    # SELECT CLASS
    class_idx = (dataset.targets == c1) | (dataset.targets == c2)
    dataset.targets = dataset.targets[class_idx]
    dataset.data = dataset.data[class_idx]
    # LOAD QUEUE
    queue = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory = True, shuffle=True, num_workers = 10)
    # LOAD MODEL
    if classifier == 'quantum':
        model = QuantumClassifier(nlayers, device).to(device)
    elif classifier == 'classical':
        model = ClassicalClassifier(nlayers, device).to(device)
    else:
        print('Choose Model: quantum or classical')
    nparams = count_parameters(model)
    nsamples = len(dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs,eta_min = 1e-6)
    # CREATE STUDY
    study = 'train__data_{}__c1_{}__c2_{}__nsamples_{}__seed_{}__classifier_{}__nparams_{}__nlayers_{}__gamma_{}'.format(data,c1,c2,nsamples,seed,classifier,nparams,nlayers,gamma)
    try:
        os.mkdir(study)
    except:
        pass
    # SSL PRETRAIN
    ssl = CreateMemoryBank(model, queue, batch_size, device)
    MemBank = ssl.initialize()
    best_loss = np.inf
    r_loss = 0.
    for epoch in range(num_epochs):
        model.train()
        for i , (inputs, _) in tqdm.tqdm(enumerate(queue)):
            model.zero_grad()
            optimizer.zero_grad()
            neg_idx = list(range(len(queue)))
            pos_idx = neg_idx.pop(i)
            x = inputs.to(device)
            x_ = transforms['random'](x)
            x = model(x)
            x_ = model(x_)
            # GET POS/NEG SAMPLES
            pos_samples = MemBank[pos_idx]
            neg_samples = [MemBank[k] for k in neg_idx]
            neg_samples = torch.cat(neg_samples, dim = 0)
            # COMPUTE KLCS LOSS
            loss = KLLoss(x, x_, pos_samples, neg_samples, gamma)
            r_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            torch.save(model.state_dict(), '{}/{}_best_model.pth'.format(study, epoch))
        print('Epoch {}|Loss: {}'.format(epoch,r_loss/len(queue)))
        if r_loss/len(queue) < best_loss:
            torch.save(model.state_dict(), '{}/best_model.pth'.format(study))
            best_loss = r_loss/len(queue)
        ssl, MemBank = None, None
        ssl = CreateMemoryBank(model, queue, batch_size, device)
        MemBank = ssl.initialize()
        r_loss = 0.
