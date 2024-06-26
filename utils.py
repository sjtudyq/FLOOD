import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
import copy
import openood

from model import *
from datasets import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData
from math import sqrt

import torch.nn as nn

import torch.optim as optim
import torchvision.utils as vutils
import time
import random

from config import params
import sklearn.datasets as sk
from sklearn.datasets import load_svmlight_file

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_svhn_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_celeba_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(datadir, split='train', target_type="attr", download=True, transform=transform)
    celeba_test_ds = CelebA_custom(datadir, split='test', target_type="attr", download=True, transform=transform)

    gender_index = celeba_train_ds.attr_names.index('Male')
    y_train =  celeba_train_ds.attr[:,gender_index:gender_index+1].reshape(-1)
    y_test = celeba_test_ds.attr[:,gender_index:gender_index+1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)

def load_femnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, download=True)
    mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, download=True)

    X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
    X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)

def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'./train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'./val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_parties, y_train, beta=0.4):
    #np.random.seed(2020)
    #torch.manual_seed(2020)

    # if dataset == 'mnist':
    #     X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    # elif dataset == 'fmnist':
    #     X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    # elif dataset == 'cifar10':
    #     X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    # elif dataset == 'svhn':
    #     X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    # elif dataset == 'celeba':
    #     X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    # elif dataset == 'femnist':
    #     X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
    # elif dataset == 'cifar100':
    #     X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    # elif dataset == 'tinyimagenet':
    #     X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    # elif dataset == 'generated':
    #     X_train, y_train = [], []
    #     for loc in range(4):
    #         for i in range(1000):
    #             p1 = random.random()
    #             p2 = random.random()
    #             p3 = random.random()
    #             if loc > 1:
    #                 p2 = -p2
    #             if loc % 2 ==1:
    #                 p3 = -p3
    #             if i % 2 == 0:
    #                 X_train.append([p1, p2, p3])
    #                 y_train.append(0)
    #             else:
    #                 X_train.append([-p1, -p2, -p3])
    #                 y_train.append(1)
    #     X_test, y_test = [], []
    #     for i in range(1000):
    #         p1 = random.random() * 2 - 1
    #         p2 = random.random() * 2 - 1
    #         p3 = random.random() * 2 - 1
    #         X_test.append([p1, p2, p3])
    #         if p1>0:
    #             y_test.append(0)
    #         else:
    #             y_test.append(1)
    #     X_train = np.array(X_train, dtype=np.float32)
    #     X_test = np.array(X_test, dtype=np.float32)
    #     y_train = np.array(y_train, dtype=np.int32)
    #     y_test = np.array(y_test, dtype=np.int64)
    #     idxs = np.linspace(0,3999,4000,dtype=np.int64)
    #     batch_idxs = np.array_split(idxs, n_parties)
    #     net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
    #     mkdirs("data/generated/")
    #     np.save("data/generated/X_train.npy",X_train)
    #     np.save("data/generated/X_test.npy",X_test)
    #     np.save("data/generated/y_train.npy",y_train)
    #     np.save("data/generated/y_test.npy",y_test)
    
    #elif dataset == 'covtype':
    #    cov_type = sk.fetch_covtype('./data')
    #    num_train = int(581012 * 0.75)
    #    idxs = np.random.permutation(581012)
    #    X_train = np.array(cov_type['data'][idxs[:num_train]], dtype=np.float32)
    #    y_train = np.array(cov_type['target'][idxs[:num_train]], dtype=np.int32) - 1
    #    X_test = np.array(cov_type['data'][idxs[num_train:]], dtype=np.float32)
    #    y_test = np.array(cov_type['target'][idxs[num_train:]], dtype=np.int32) - 1
    #    mkdirs("data/generated/")
    #    np.save("data/generated/X_train.npy",X_train)
    #    np.save("data/generated/X_test.npy",X_test)
    #    np.save("data/generated/y_train.npy",y_train)
    #    np.save("data/generated/y_test.npy",y_test)

    # elif dataset in ('rcv1', 'SUSY', 'covtype'):
    #     X_train, y_train = load_svmlight_file(datadir+dataset)
    #     X_train = X_train.todense()
    #     num_train = int(X_train.shape[0] * 0.75)
    #     if dataset == 'covtype':
    #         y_train = y_train-1
    #     else:
    #         y_train = (y_train+1)/2
    #     idxs = np.random.permutation(X_train.shape[0])

    #     X_test = np.array(X_train[idxs[num_train:]], dtype=np.float32)
    #     y_test = np.array(y_train[idxs[num_train:]], dtype=np.int32)
    #     X_train = np.array(X_train[idxs[:num_train]], dtype=np.float32)
    #     y_train = np.array(y_train[idxs[:num_train]], dtype=np.int32)

    #     mkdirs("data/generated/")
    #     np.save("data/generated/X_train.npy",X_train)
    #     np.save("data/generated/X_test.npy",X_test)
    #     np.save("data/generated/y_train.npy",y_train)
    #     np.save("data/generated/y_test.npy",y_test)

    # elif dataset in ('a9a'):
    #     X_train, y_train = load_svmlight_file(datadir+"a9a")
    #     X_test, y_test = load_svmlight_file(datadir+"a9a.t")
    #     X_train = X_train.todense()
    #     X_test = X_test.todense()
    #     X_test = np.c_[X_test, np.zeros((len(y_test), X_train.shape[1] - np.size(X_test[0, :])))]

    #     X_train = np.array(X_train, dtype=np.float32)
    #     X_test = np.array(X_test, dtype=np.float32)
    #     y_train = (y_train+1)/2
    #     y_test = (y_test+1)/2
    #     y_train = np.array(y_train, dtype=np.int32)
    #     y_test = np.array(y_test, dtype=np.int32)

    #     mkdirs("data/generated/")
    #     np.save("data/generated/X_train.npy",X_train)
    #     np.save("data/generated/X_test.npy",X_test)
    #     np.save("data/generated/y_train.npy",y_train)
    #     np.save("data/generated/y_test.npy",y_test)


    n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200

        N = y_train.shape[0]
        #np.random.seed(2020)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break


        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if dataset == "cifar100":
            K = 100
        elif dataset == "tinyimagenet":
            K = 200
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(K)]
            contain=[]
            for i in range(n_parties):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1

    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        
    elif partition == "mixed":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        times=[1 for i in range(10)]
        contain=[]
        for i in range(n_parties):
            current=[i%K]
            j=1
            while (j<2):
                ind=random.randint(0,K-1)
                if (ind not in current and times[ind]<2):
                    j=j+1
                    current.append(ind)
                    times[ind]+=1
            contain.append(current)
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
        

        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*n_train)

        for i in range(K):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)

            proportions_k = np.random.dirichlet(np.repeat(beta, 2))
            #proportions_k = np.ndarray(0,dtype=np.float64)
            #for j in range(n_parties):
            #    if i in contain[j]:
            #        proportions_k=np.append(proportions_k ,proportions[j])

            proportions_k = (np.cumsum(proportions_k)*len(idx_k)).astype(int)[:-1]

            split = np.split(idx_k, proportions_k)
            ids=0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                    ids+=1

    elif partition == "real" and dataset == "femnist":
        num_user = u_train.shape[0]
        user = np.zeros(num_user+1,dtype=np.int32)
        for i in range(1,num_user+1):
            user[i] = user[i-1] + u_train[i-1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, n_parties)
        net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(n_parties)}
        for i in range(n_parties):
            for j in batch_idxs[i]:
                net_dataidx_map[i]=np.append(net_dataidx_map[i], np.arange(user[j], user[j+1]))
                
    elif partition == "transfer-from-femnist":
        stat = np.load("femnist-dis.npy")
        n_total = stat.shape[0]
        chosen = np.random.permutation(n_total)[:n_parties]
        stat = stat[chosen,:]
        
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
        else:
            K = 10
        
        N = y_train.shape[0]
        #np.random.seed(2020)
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = stat[:,k]
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
  

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "transfer-from-criteo":
        stat0 = np.load("criteo-dis.npy")
        
        n_total = stat0.shape[0]
        flag=True
        while (flag):
            chosen = np.random.permutation(n_total)[:n_parties]
            stat = stat0[chosen,:]
            check = [0 for i in range(10)]
            for ele in stat:
                for j in range(10):
                    if ele[j]>0:
                        check[j]=1
            flag=False
            for i in range(10):
                if check[i]==0:
                    flag=True
                    break
                    
        
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            stat[:,0]=np.sum(stat[:,:5],axis=1)
            stat[:,1]=np.sum(stat[:,5:],axis=1)
        else:
            K = 10
        
        N = y_train.shape[0]
        #np.random.seed(2020)
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = stat[:,k]
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
  

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (net_dataidx_map, traindata_cls_counts)


def get_trainable_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    # logger.info("net.parameter.data:", list(net.parameters()))
    paramlist=list(trainable)
    N=0
    for params in paramlist:
        N+=params.numel()
        # logger.info("params.data:", params.data)
    X=torch.empty(N,dtype=torch.float64)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel
    # logger.info("get trainable x:", X)
    return X


def put_trainable_parameters(net,X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel

def compute_accuracy_vote(model_list, threshold_list, dataloader, accepted_vote, normalize = True, factor=1, mode=1, device="cpu"):
    for model in model_list:
        model.eval()
        model.to(device)

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, sample in enumerate(tmp):
                x, target = sample['data'].to(device), sample['label'].to(device)
                out = [model(x)[0].cpu() for model in model_list]
                for i in range(len(out)):
                    #out_del = out[i].numpy()
                    #out_max = np.repeat(np.max(out_del[:,:-1], axis=1), out_del.shape[1]).reshape(-1, out_del.shape[1])
                    #out_del = np.where(out_del > out_max - 1e-5 ,-10000, out_del)
                    #out_del = torch.softmax(torch.Tensor(out_del), dim=1).numpy()
                    
                    #confidence = out_del[:,-1]
                    
                    saved = torch.softmax(out[i][:,:-1], dim=1).numpy()
                    out[i] = torch.softmax(out[i], dim=1).numpy()

                    #out[i][:,:-1] = saved # new added, just calculate existing class

                    if normalize:
                        out[i][:,:-1] = saved
                        
                        #out[i][:,:-1] *= np.repeat(confidence,out[i].shape[1]-1).reshape(-1, out[i].shape[1]-1)
                        #out[i][:,-1] = -np.max(out[i][:,:-1], axis=1)
                        out[i][:,-1] = (np.log(out[i][:,-1]) - threshold_list[i][0]) / (threshold_list[i][mode] - threshold_list[i][0])
                        out[i][:,-1] = np.where(out[i][:,-1]<0, 0, out[i][:,-1])
                        out[i][:,-1] = np.where(out[i][:,-1]>1, 1, out[i][:,-1])
                        out[i][:,:-1] = out[i][:,:-1] - out[i][:,:-1] * np.repeat(out[i][:,-1],out[i].shape[1]-1).reshape(-1, out[i].shape[1]-1)
                    out[i] = out[i] ** factor
                    out[i] = out[i].tolist()
                pred_label = []
                for ind in range(len(out[0])):
                    vote = [result[ind] for result in out]
                    vote = np.array(vote)
                    index = np.argsort(vote[:,-1])
                    sorted_vote = vote[index]
                    final_vote = np.sum(sorted_vote[:accepted_vote, :-1], axis=0)
                    pred = int(np.argmax(final_vote))
                    pred_label.append(pred)
                    '''
                    if batch_idx == 0:
                        logger.info(target[ind])
                        logger.info(pred)
                        logger.info(sorted_vote)
                    '''
                pred_label = torch.LongTensor(pred_label).to(device)


                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            half = int(batch_idx / 2)

    return correct/float(total), half, pred_labels_list

def compute_accuracy(model, dataloader, get_confusion_matrix=False, calc=False, device="cpu", add=0):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    outlier_prob, num = 0.0, 0
    max_prob = 0
    avg_max, avg_num = 0.0, 0
    max_tmp = 0
    flag = False
    ftrs = None
    lbs = None
    score_list = []
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, sample in enumerate(tmp):
                x, target = sample['data'].to(device), sample['label'].to(device)
                out = model(x)
                '''
                if not flag:
                    ftrs = mid.cpu().numpy()
                    lbs = target.cpu().numpy()
                    flag = True
                else:
                    ftrs = np.concatenate((ftrs,mid.cpu().numpy()))
                    lbs = np.concatenate((lbs,target.cpu().numpy()))
                '''
                prob = torch.softmax(out, dim=1)
                # if batch_idx==0:
                #     logger.info(prob)
                #     logger.info(target)
                if calc:
                    score = -prob[:,-1]
                    score_list.append(score)

                        
                _, pred_label = torch.max(out.data, 1)

                
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())


    ''''
    if not calc:        
        ftrs = np.concatenate((ftrs,add))
        lbs = np.concatenate((lbs,np.ones(add.shape[0],dtype=np.int32)*10))

        tsne = TSNE()
        result = tsne.fit_transform(ftrs)
        np.save('ft.npy',result)
        np.save('lb.npy',lbs)
    '''
    #if get_confusion_matrix:
    #    conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    #if get_confusion_matrix:
    #    return correct/float(total), conf_matrix

    if calc:
        score_all = torch.cat(score_list)
        k = int(len(score_all)*0.05)
        threshold_value, _ = torch.kthvalue(score_all, k)
        return correct/float(total), threshold_value #outlier_prob / num, torch.log(max_prob), avg_max / avg_num
    else:
        return correct/float(total)


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataloader(config):
    from openood.datasets import get_dataloader as gd
    loader_dict = gd(config)
    train_loader = loader_dict['train']
    test_loader = loader_dict['test']
    return train_loader, test_loader


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def noise_sample(choice, n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        c_tmp = np.array(choice)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(len(choice), size=batch_size)
            for j in range(batch_size):
                idx[i][j] = c_tmp[int(idx[i][j])]

            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx

def compute_auc_outlier_detection(inlier, model, dataloader, get_confusion_matrix=False, calc=False, device="cpu", add=0):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    outlier_prob, num = 0.0, 0
    max_prob = 0
    avg_max, avg_num = 0.0, 0
    max_tmp = 0
    flag = False
    ftrs = None
    lbs = None
    

    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, sample in enumerate(tmp):
                x, target = sample['data'].to(device), sample['label'].to(device)
                out = model(x)

                prob = torch.softmax(out, dim=1)

                target_revised = np.where(target.cpu().numpy()==inlier, 0, 1)
                
                if not flag:
                    ftrs = prob[:,10].cpu().numpy() # record outlier prob instead
                    lbs = target_revised
                    flag = True
                else:
                    ftrs = np.concatenate((ftrs,prob[:,10].cpu().numpy()))
                    lbs = np.concatenate((lbs,target_revised))

                
                if calc:
                    if torch.max(prob[:,-1]) > max_prob:
                        max_prob = torch.max(prob[:,-1])

                    if torch.sum(torch.log(prob[:,-1])) > -10000:
                        outlier_prob += torch.sum(torch.log(prob[:,-1]))
                        num += x.shape[0]

                    if torch.max(prob[:,-1]) > max_tmp:
                        max_tmp = torch.max(prob[:,-1])

                    if batch_idx % 4 == 0:
                        avg_max += torch.log(max_tmp)
                        avg_num += 1
                        max_tmp = 0
                        
                _, pred_label = torch.max(out.data, 1)

                #if batch_idx == 0:
                #    logger.info(out.data)
                #    logger.info(target)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
    ''''
    if not calc:        
        ftrs = np.concatenate((ftrs,add))
        lbs = np.concatenate((lbs,np.ones(add.shape[0],dtype=np.int32)*10))

        tsne = TSNE()
        result = tsne.fit_transform(ftrs)
        np.save('ft.npy',result)
        np.save('lb.npy',lbs)
    '''
    #if get_confusion_matrix:
    #    conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    #if get_confusion_matrix:
    #    return correct/float(total), conf_matrix
    #logger.info(lbs)
    #logger.info(ftrs)
    auc = roc_auc_score(lbs, ftrs)

    return auc

def distill(model, first_half_labels, dataloader, half, args, device="cpu"):
    model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(100):
        epoch_loss_collector = []

        for batch_idx, (x, target) in enumerate(dataloader):
            if batch_idx >= half:
                break
            bs = target.shape[0]
            target = torch.Tensor(first_half_labels[bs*batch_idx:bs*(batch_idx+1)])
            x, target = x.to(device), target.to(device)
            
            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out, mid = model(x) 

            loss = criterion(out, target) 
            
            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if batch_idx < half:
                continue
            x, target = x.to(device), target.to(device,dtype=torch.int64)
            out, mid = model(x)
                
            prob = torch.softmax(out, dim=1)
                    
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    logger.info(correct/float(total))
    
    
def compute_accuracy_vote_soft(model_list, threshold_list, dataloader, accepted_vote, normalize = True, factor=1, mode=1, device="cpu"):
    for model in model_list:
        model.eval()
        #model.to(device)

    true_labels_list, pred_labels_list = np.array([]), np.array([[0 for i in range(10)]])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    out_total = [[] for i in range(1000)]
    for i in range(len(model_list)):
        model = model_list[i].to(device)
        with torch.no_grad():
            for tmp in dataloader:
                for batch_idx, sample in enumerate(tmp):
                    x, target = sample['data'].to(device), sample['label'].to(device)
                    logits = model(x).cpu()
                    out_total[batch_idx].append(logits)
        #logger.info(batch_idx)
        model.to('cpu')

    pred_labels_list = np.array([[0 for i in range(out_total[0][0].shape[1]-1)]])

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, sample in enumerate(tmp):
                x, target = sample['data'].to(device), sample['label'].to(device)
                out = out_total[batch_idx]
                #out = [model(x)[0].cpu() for model in model_list]
                for i in range(len(out)):
                    #out_del = out[i].numpy()
                    #out_max = np.repeat(np.max(out_del[:,:-1], axis=1), out_del.shape[1]).reshape(-1, out_del.shape[1])
                    #out_del = np.where(out_del > out_max - 1e-5 ,-10000, out_del)
                    #out_del = torch.softmax(torch.Tensor(out_del), dim=1).numpy()
                    
                    #confidence = out_del[:,-1]
                    out[i] = torch.softmax(out[i][:,:], dim=1)
                    if normalize:
                        score = -out[i][:,-1]
                        mask = torch.where(score>threshold_list[i],1,0)
                        out[i] = out[i] * mask.reshape(-1,1)
                    
                    #saved = torch.softmax(out[i][:,:-1], dim=1).numpy()
                    #out[i] = out[i] / len(model_list)

                    #out[i][:,:-1] = saved # new added, just calculate existing class

                    out[i] = out[i].tolist()
                pred_label = []
                prob_list = []
                for ind in range(len(out[0])):
                    vote = [result[ind] for result in out]
                    vote = np.array(vote)
                    final_vote = np.sum(vote[:, :-1], axis=0)
                    #probob = torch.softmax(torch.Tensor(final_vote), dim=0).tolist()

                    probob = (final_vote / np.sum(final_vote)).tolist()

                    prob_list.append(probob)
                    pred = int(np.argmax(final_vote))
                    pred_label.append(pred)
                    '''
                    if batch_idx == 0:
                        logger.info(target[ind])
                        logger.info(pred)
                        logger.info(sorted_vote)
                    '''
                pred_label = torch.LongTensor(pred_label).to(device)
                prob_list = torch.Tensor(prob_list).to(device)


                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, prob_list.numpy(), axis=0)
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, prob_list.cpu().numpy(), axis=0)
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

            half = int(batch_idx / 2)

    return correct/float(total), half, pred_labels_list[1:]

def distill_soft(model, first_half_labels, dataloader, half, args, device="cpu"):
    model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    criterion = nn.KLDivLoss().to(device)

    for epoch in range(100):
        epoch_loss_collector = []

        for batch_idx, (x, target) in enumerate(dataloader):
            if batch_idx >= half:
                break
            bs = target.shape[0]
            target = torch.Tensor(first_half_labels[bs*batch_idx:bs*(batch_idx+1)])
            x, target = x.to(device), target.to(device)
            
            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            #target = target.long()

            out, mid = model(x) 
            out = torch.nn.LogSoftmax(dim=1)(out[:,:-1])

            loss = criterion(out, target) 
            
            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if batch_idx < half:
                continue
            x, target = x.to(device), target.to(device,dtype=torch.int64)
            out, mid = model(x)
                
            prob = torch.softmax(out[:,:-1], dim=1)
                    
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    logger.info(correct/float(total))
