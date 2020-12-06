import config
######################  fine tune params ###############################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dropout_resnet', type=float,default=config.dropout_resnet)
parser.add_argument('--dropout_linear', type=float, default=config.dropout_linear)
parser.add_argument('--pooling_size_resnet',type=int, default=config.pooling_size_resnet)
parser.add_argument('--resnet_layers', nargs='+', type=int,default=config.resnet_layers)
parser.add_argument('--batchsize',type=int, default=config.batchsize)
parser.add_argument('--crop_size',type=int, default=config.crop_size)
parser.add_argument('--train_path',type=str, default=config.train_path)
parser.add_argument('--init_xavri',type=str, default=config.init_xavri)
parser.add_argument('--SGD',type=str, default=config.SGD)

args = parser.parse_args()

print(args.init_xavri)

config.dropout_resnet = args.dropout_resnet
config.dropout_linear = args.dropout_linear
config.pooling_size_resnet = args.pooling_size_resnet
config.resnet_layers = args.resnet_layers
config.batchsize = args.batchsize
# config.batchsize = 12
config.crop_size = args.crop_size
config.train_path = args.train_path
config.init_xavri = args.init_xavri
config.SGD = args.SGD

print("---------------PARAMS--------------------")
print("dropout_resnet:",config.dropout_resnet)
print("dropout_linear:",config.dropout_linear)
print("pooling_size_resnet:",config.pooling_size_resnet)
print("resnet_layers:",config.resnet_layers)
print("batchsize:",config.batchsize)
print("crop_size:",config.crop_size)
print("init_xavri:",config.init_xavri)
print("SGD:",config.SGD)
print("-----------------------------------------")

########################################################################
import torch
import numpy as np
import random
# 设置随机数种子
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
random.seed(seed)
np.random.seed(seed)
import torch.nn as nn
import os
from dataLoader import get_loader
import pandas as pd
import time
from utilize import utilize
from tqdm import tqdm
#########################################
# 在这选模型
from layers.resnet50 import model
from layers.capsnet import model
from layers.densenet import model
from layers.dpn import model
from layers.resnet_inception import model
from layers.resnext import model
from layers.transformer import model

import torch.optim.lr_scheduler as lr_scheduler

import warnings
warnings.filterwarnings("ignore")

# cuda设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)

# Create model directory
if not os.path.exists(config.model_path):
    os.makedirs(config.model_path)

# 建立模型
print("construct model...")
net = model().to(device)

# net = model().to(device)
if config.pre_train:
    utilize.load_pre_model(net, config.pre_model)

# Loss and optimizer
from layers.FocalLoss import FocalLoss
criterion = FocalLoss()

# for name,parameters in net.named_parameters():
#   print(name,':',parameters.size())
# optimizer = torch.optim.Adam([{'params':[ param for name, param in net.named_parameters() if 'linear'  in name]}], lr=config.lr)
if config.SGD == "True":
    # optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, weight_decay=0.05)
    # optimizer = adabound.AdaBound(net.parameters(), lr=config.lr, final_lr=0.1)
    print("SGD")
else:
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr,  weight_decay=0.05)


#根据式子进行计算
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,eta_min=5e-6,)

# 设置记录文件
log_txt = config.log_path + "epoch_loss_lr_" + str(config.lr) + "_batchsize_" + str(config.batchsize) + ".txt"
utilize.log_txt(filename=log_txt, mark=True)

# 加载数据
train_loader = get_loader(train="train")
total_step = len(train_loader)
eval_loader = get_loader(train="val")
eval_step = len(eval_loader)

log_info = []

print("train start...")
best_score = 0
for epoch in range(config.num_epochs):
    loss_epoch = 0

    net.train()
    t1 = time.time()
    # for i,(af,vf,  label,name) in enumerate(train_loader):
    for i, ( vf, label, name) in enumerate(train_loader):

        vf = vf.to(device)
        # af = af.to(device)
        label = label.to(device)
        # singer = singer.to(device)


        # out,_ = net(vf)
        out,_ = net(vf)

        # print(out.shape, label.shape, label)
        # print(out)
        loss = criterion(out, label)

        net.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

        # Print log info
        if i % config.log_step == (config.log_step - 1):
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, config.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
    t2 = time.time()

    # Save the model checkpoints
    torch.save(net.state_dict(), os.path.join(config.model_path, str(epoch)+'_net.ckpt'))

    loss_epoch = loss_epoch / total_step

    # 调整学习率
    utilize.adjust_learning_rate(optimizer, epoch, config.lr, config.lr_change_rate, config.lr_change_epoch)
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    # scheduler.step()
    # lr = scheduler.get_lr()
    # print(lr)

    # 验证
    if epoch % config.eval_epoch == (config.eval_epoch-1):
        print("Start eval ...")
        net.eval()

        accuracy = 0
        # for i, (af,vf, label,name) in enumerate(eval_loader):
        for i, ( vf, label, name) in enumerate(eval_loader):

            # af = af.to(device)
            vf = vf.to(device)
            # out,_ , s= net(vf)
            out,_= net(vf)


            pred_y = torch.max(out, 1)[1].data.cpu().numpy()
            # print(pred_y)
            label = label.data.numpy()
            # print(label)

            accuracy += sum(np.equal(pred_y,label))

        accuracy = (accuracy*1.0/eval_step)/config.batchsize
        print(accuracy)


    # 记录
    utilize.log_txt(filename=log_txt,contexts="epoch:"+str(epoch)+"  time:"+str(t2-t1)+"s  loss:"+str(loss_epoch)+"  acc:"+str(accuracy)+"\n")
    # 保存成npy可视化
    log_info.append([epoch,t2-t1, loss_epoch, accuracy])
    np.save("logs/logs.npy", log_info, allow_pickle=True)

    if accuracy>best_score:
        # Save the model checkpoints
        torch.save(net.state_dict(), os.path.join(config.model_path, 'best.ckpt'))
        best_score = accuracy