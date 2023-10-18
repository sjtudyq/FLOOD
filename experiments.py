import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
import copy
from PIL import Image
from cutpaste import *

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from attack import *
import openood
from openood.utils import setup_config
import main

def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--config', dest='config', nargs='+', required=True)
    parser.add_argument('--mark', type=str, default='default', help='OOD algorithm name')
    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if args.dataset == "generated":
            net = PerceptronModel()
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16,8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
        elif args.model == "vgg":
            net = vgg11()
        elif args.model == "lenet":
            net = LeNet()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=11)
            elif args.dataset in ("cifar100"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=101)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=11)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        elif args.model == "resnet":
            if args.dataset == "cifar100":
                net = ResNet18_cifar10(num_classes=101)
            elif args.dataset == "tinyimagenet":
                net = ResNet18_cifar10(num_classes=201)
            else:
                net = ResNet18_cifar10(num_classes=11)
        elif args.model == "vgg16":
            net = vgg16()
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

op = transforms.RandomChoice( [
    #transforms.RandomResizedCrop(sz),
    transforms.RandomRotation(degrees=(15,75)),
    transforms.RandomRotation(degrees=(-75,-15)),
    transforms.RandomRotation(degrees=(85,90)),
    transforms.RandomRotation(degrees=(-90,-85)),
    transforms.RandomRotation(degrees=(175,180)),
    #transforms.RandomAffine(0,translate=(0.2,0.2)),
    #transforms.RandomPerspective(distortion_scale=1,p=1),
    #transforms.RandomHorizontalFlip(p=1),
    #transforms.RandomVerticalFlip(p=1)
])

def cut(x):
    x_gen = copy.deepcopy(x.cpu().numpy())
    half = int(x_gen.shape[2] / 2)
    rnd = random.randint(0,5)
    pl = random.randint(0,half-1)
    pl2 = random.randint(0,half-1)
    while (abs(pl-pl2)<half/2):
        pl2 = random.randint(0,half-1)
    if rnd <= 1:
        x_gen[:,:,pl:pl+half] = x_gen[:,:,pl2:pl2+half]
    elif rnd == 2:
        x_gen[:,:,half:] = x_gen[:,:,:half]
        x_gen[:,:,:half] = copy.deepcopy(x.cpu().numpy())[:,:,half:]
    elif rnd <= 4:
        x_gen[:,pl:pl+half,:] = x_gen[:,pl2:pl2+half,:]
    else:
        x_gen[:,half:,:] = x_gen[:,:half,:]
        x_gen[:,:half,:] = copy.deepcopy(x.cpu().numpy())[:,half:,:]
    x_gen = torch.Tensor(x_gen)

    return x_gen

def rot(x):
    #rnd = random.randint(0,20)
    #if rnd < 21:
    x_gen = copy.deepcopy(x.cpu().numpy())
    half = int(x_gen.shape[2] / 2)
    pl = random.randint(0,half-1)
    rnd = random.randint(1,3)

    x_gen[:,pl:pl+half,half:] = np.rot90(x_gen[:,pl:pl+half,half:],k=rnd,axes=(1,2))
    x_gen[:,pl:pl+half,:half] = np.rot90(x_gen[:,pl:pl+half,:half],k=rnd,axes=(1,2))
    x_gen = torch.Tensor(x_gen)
    #else:
    #    x_gen = op(copy.deepcopy(x))
    #    if rnd < 20:
    #        x_gen = torch.max(x_gen, x)
    #    else:
    #        x_gen = torch.min(x_gen, x)

    return x_gen

def paint(x):
    x_gen = copy.deepcopy(x.cpu().numpy())
    size = int(x_gen.shape[2])
    sq = 4
    pl = random.randint(sq,size-sq*2)
    pl2 = random.randint(sq,size-sq-1)
    rnd = random.randint(0,1)
    if rnd == 0:
        for i in range(sq,size-sq):
            x_gen[:,i,pl:pl+sq] = x_gen[:,pl2,pl:pl+sq]
    elif rnd == 1:
        for i in range(sq,size-sq):
            x_gen[:,pl:pl+sq,i] = x_gen[:,pl:pl+sq,pl2]
    x_gen = torch.Tensor(x_gen)

    return x_gen

def blur(x):
    rnd = random.randint(0,1)
    sz = random.randint(1,4)*2+1
    sz2 = random.randint(0,2)*2+1
    if rnd == 0:
        func = transforms.GaussianBlur(kernel_size=(sz, sz2), sigma=(10, 100))
    else:
        func = transforms.GaussianBlur(kernel_size=(sz2, sz), sigma=(10, 100))
    
    return func(x)

def shuffle(x):
    rnd = random.randint(0,1)
    x_gen = copy.deepcopy(x.cpu().numpy())
    sz = x_gen.shape[0]
    li = np.split(x_gen, range(1,sz,10), axis=rnd)
    np.random.shuffle(li)
    t = np.concatenate(li, axis=rnd)
    t = torch.Tensor(t)
    return t

def train_net_vote(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, sz, num_class=10, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
 
    criterion = nn.CrossEntropyLoss(ignore_index=11).to(device)
    #c2 = nn.CrossEntropyLoss(reduction='none').to(device)
    net.to(device)

    cnt = 0
    rnd = []

    toImg = transforms.ToPILImage()
    toTensor = transforms.ToTensor()

    op = transforms.RandomChoice( [
        #transforms.RandomResizedCrop(sz),
        transforms.RandomRotation(degrees=(15,75)),
        transforms.RandomRotation(degrees=(-75,-15)),
        transforms.RandomRotation(degrees=(85,90)),
        transforms.RandomRotation(degrees=(-90,-85)),
        transforms.RandomRotation(degrees=(175,180)),
        #transforms.RandomAffine(0,translate=(0.2,0.2)),
        #transforms.RandomPerspective(distortion_scale=1,p=1),
        #transforms.RandomHorizontalFlip(p=1),
        #transforms.RandomVerticalFlip(p=1)
    ])

    aug = transforms.Compose([
        toImg,
        op,
        toTensor
    ])

    aug_crop =  transforms.RandomChoice( [
        transforms.RandomResizedCrop(sz, scale=(0.1, 0.33)), # good
        transforms.Lambda(lambda img: blur(img)), # good
        #transforms.Lambda(lambda img: shuffle(img)), # bad
        transforms.RandomErasing(p=1, scale=(0.33, 0.5)), # good
        transforms.Lambda(lambda img: cut(img)), # fine
        transforms.Lambda(lambda img: rot(img)),
        transforms.Lambda(lambda img: cut(img)),
        transforms.Lambda(lambda img: rot(img)),
        #transforms.Lambda(lambda img: paint(img))
    ])

    attack = FastGradientSignUntargeted(net, 
                                        epsilon=0.5, 
                                        alpha=0.002, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=5,
                                        device=device)

    cp = CutPasteUnion()

    aug_final = transforms.RandomChoice( [
        transforms.Lambda(lambda img: aug_crop(img)),
        #transforms.Lambda(lambda img: cp(img)) # delete this comment if you want to add cutpaste augmentation
    ])

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, sample in enumerate(train_dataloader):

            x, target = sample['data'].to(device), sample['label'].to(device)
            y_gen = np.ones(x.shape[0]) * num_class
            y_gen = torch.LongTensor(y_gen).to(device)
         
            x_gen11 = copy.deepcopy(x.cpu().numpy())
            for i in range(x_gen11.shape[0]):
                x_gen11[i] = aug_final(torch.Tensor(x_gen11[i]))
            x_gen11 = torch.Tensor(x_gen11).to(device)
            
            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()
            y_gen.requires_grad = False
          
            x_gen11.requires_grad = True

            # For MNIST:
            if args.dataset == "mnist":
                out, mid = net(x, return_feature=True) 
            
                out_gen11 = net(x_gen11)
                
                one_hot = torch.zeros(out.cpu().shape[0], out.cpu().shape[1]).scatter_(1, target.cpu().reshape(-1, 1), 1)
                one_hot = one_hot.to(device)
                out_second = out - one_hot * 10000

                ind = np.arange(x.shape[0])
                np.random.shuffle(ind)
                y_mask = np.arange(x.shape[0])
                labels = target.cpu().numpy()
                y_mask = np.where(labels[y_mask] == labels[ind[y_mask]], 11, 10)
                
                loss = criterion(out, target) + criterion(out_gen11, y_gen) + criterion(out_second, y_gen) 

                if np.min(y_mask) == 10:                    
                    y_mask = torch.LongTensor(y_mask).to(device)

                    beta=torch.distributions.beta.Beta(1, 1).sample([]).item()
                    mixed_embeddings = beta * mid + (1-beta) * mid[ind]
                    mixed_out = net.later_layers(mixed_embeddings) 
                    loss += criterion(mixed_out, y_mask) * 0.01
                
                adv_data = attack.perturb(x_gen11, y_gen)

                out_adv = net(adv_data)

                loss += criterion(out_adv, y_gen)
            else:

                # For CIFAR-10 and CIFAR-100, ResNet:
                x_con = torch.cat([x,x_gen11],dim=0)
                y_con = torch.cat([target,y_gen],dim=0)
                out = net(x_con)
                loss = criterion(out, y_con)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        #auc = compute_auc_outlier_detection(net_id, net, test_dataloader, device=device) #can be used to perform traditional outlier detection experiments, calculate ROC-AUC under noniid-#label1 partition
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        #train_acc = compute_accuracy(net, train_dataloader, device=device)
        #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        #writer.add_scalar('Accuracy/train', train_acc, epoch)
        #writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

        #device = 'cpu'
        #net.to(device)
        '''
        flag = False
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x_gen11 = copy.deepcopy(x.cpu().numpy())
                for i in range(x_gen11.shape[0]):
                    x_gen11[i] = aug_crop(torch.Tensor(x_gen11[i]))
                x_gen11 = torch.Tensor(x_gen11).to(device)

                out, mid = net(x_gen11)

                if not flag:
                    flag = True
                    outliers = mid.cpu().detach().numpy()
                else:
                    outliers = np.concatenate((outliers,mid.cpu().detach().numpy()))
        '''
    train_acc, threshold= compute_accuracy(net, train_dataloader, calc=True, device=device)
    test_acc = compute_accuracy(net, test_dataloader, device=device)#, add=outliers)
    
    logger.info(threshold)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    logger.info(' ** Training complete **')
    return threshold


def local_train_net_vote(nets, selected, args, net_dataidx_map, config, partition_folder_name, test_dl = None, device="cpu"):
    threshold_list = []

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        config.dataset['train'].imglist_pth = '%s%d.txt'%(partition_folder_name,net_id)

        train_dl_local, test_dl = get_dataloader(config)
        n_epoch = args.epochs

        if args.dataset in ('mnist', 'fmnist'):
            sz = 28
        else:
            sz = 32
        
        num_class = 10
        if args.dataset == 'cifar100':
            num_class = 100
        elif args.dataset == 'tinyimagenet':
            num_class = 200
        
        threshold = train_net_vote(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, sz, num_class=num_class, device=device)
        threshold_list.append(float(threshold))
       
    return threshold_list


if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Partitioning data")

    train_file_name = "data_info/train_%s.txt"%(args.dataset)
    test_file_name = "data_info/test_%s.txt"%(args.dataset)
    partition_folder_name = "data_info/%s_%s_%s_%s/"%(args.dataset,args.partition,args.n_parties,args.init_seed)
        
    try:
        os.makedirs(partition_folder_name)
    except:
        pass

    data_loc = []
    data_y = []

    # Open the file for reading
    with open(train_file_name, "r") as file:
        # Read each line from the file
        for line in file:
            # Split the line into string and number using space as a delimiter
            parts = line.split()
            if len(parts) == 2:
                string_value = parts[0]
                number_value = int(parts[1])  # Convert the number to an integer
                data_loc.append(string_value)
                data_y.append(number_value)

    y_train = np.array(data_y)
    net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, y_train, beta=args.beta)

    for net_id in range(args.n_parties):
        split_file_name = '%s%d.txt'%(partition_folder_name, net_id)
        with open(split_file_name, "w") as file:
            for ind in net_dataidx_map[net_id]:
                file.write("%s %d\n"%(data_loc[ind], data_y[ind]))

            

    n_classes = len(np.unique(y_train))

    config = setup_config()
    train_dl_global, test_dl_global = get_dataloader(config)

    print("len train_dl_global:", len(train_dl_global))


    data_size = len(test_dl_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)


    if args.alg == 'fedov':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        threshold_list=[]
        threshold_list = local_train_net_vote(nets, arr, args, net_dataidx_map, config, partition_folder_name, test_dl = test_dl_global, device=device)
        #logger.info(threshold_list)
        model_list = [net for net_id, net in nets.items()]
        for factor in [1]:
            logger.info("Factor = {}".format(factor))
            #logger.info("Normalize")
            #for accepted_vote in range(1, 11):
            #    test_acc = compute_accuracy_vote(model_list, threshold_list, test_dl_global, accepted_vote, factor=factor,device=device)
            #    logger.info("Max {} vote: test acc = {}".format(accepted_vote, test_acc))
            
            logger.info("Not Normalize")
            for accepted_vote in range(10, 11):
                test_acc, half, pred_labels_list = compute_accuracy_vote_soft(model_list, threshold_list, test_dl_global, accepted_vote, normalize = True, factor=factor,device=device)
                logger.info("Max {} vote: test acc = {}".format(accepted_vote, test_acc))
    else:

        # config.optimizer.num_epochs = 1

        folder_name = "/ssd1/yufeng/saved_model/%s_%s_%s_%s_%s/"%(args.dataset,args.partition,args.n_parties,args.init_seed, args.mark)
        for net_id in range(args.n_parties):
            dataidxs = net_dataidx_map[net_id]

            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

            config.dataset['train'].imglist_pth = '%s%d.txt'%(partition_folder_name,net_id)
            client_folder_name = '%s%s/'%(folder_name,net_id)
            try:
                os.makedirs(client_folder_name)
            except:
                pass
            config.output_dir = client_folder_name

            train_dl_local, test_dl = get_dataloader(config)
            openood.utils.launch(
                main.main,
                config.num_gpus,
                num_machines=config.num_machines,
                machine_rank=config.machine_rank,
                dist_url='auto',
                args=(config, ),
            )
        
        from openood.networks import get_network
        model_list = []
        for net_id in range(args.n_parties):
            net = get_network(config.network)
            client_folder_name = '%s%s/'%(folder_name,net_id)
            net.load_state_dict(torch.load('%slast.pth'%(client_folder_name)))
            model_list.append(net)

        from openood.pipelines.test_ood_pipeline import TestFLOODPipeline
        total_result = []
        for net_id in range(args.n_parties):
            config.dataset['train'].imglist_pth = '%s%d.txt'%(partition_folder_name,net_id)
            test_pipeline = TestFLOODPipeline(config)
            result = test_pipeline.run(model_list[net_id])
            total_result.append(result)
        total, correct = 0, 0
        for batch_id in range(len(total_result[0])):
            target = total_result[0][batch_id]['target'].to(device)
            num = target.shape[0]
            total += num
            vote = torch.zeros((num, config.dataset.num_classes)).to(device)
            for net_id in range(args.n_parties):
                pred = total_result[net_id][batch_id]['pred'].to(device)
                conf = total_result[net_id][batch_id]['conf'].to(device)
                for data_id in range(len(conf)):
                    vote[data_id][pred[data_id]] += conf[data_id]
            pred = vote.max(1)[1]
            correct += pred.eq(target.data).sum().item()
        logger.info("test acc = {}".format(correct/total))
        print("test acc = {}".format(correct/total))

    os.system("rm -r %s"%(folder_name))
    #train_acc = compute_accuracy_vote(nets, train_dl_global)
    
        #logger.info(half)
        #logger.info(pred_labels_list.shape)
        #logger.info(pred_labels_list)

    # stu_nets = init_nets(args.net_config, args.dropout_p, 1, args)
    # stu_model = stu_nets[0][0]
    # distill_soft(stu_model, pred_labels_list, test_dl_global, half, args=args, device=device)
    # compute_accuracy_vote_soft() and distill_soft() for soft label distillation like FedDF. 
    # compute_accuracy_vote() and distill() are hard label distillation.
    # Soft label is usually better, especially for complicated datasets like CIFAR-10, CIFAR-100.
        
