# coding:utf-8

import torch
import torch.nn as nn
import re
# Only for this densenet
def load_pre_model(net, pre_model):
    try:
        print("loading model...")
        net.load_state_dict(torch.load(pre_model))
    except:
        pre_net = torch.load(pre_model)
        remove_list = [
            "query_lstm","feature_lstm","classlabel_lstm","fc"
        ]
        for k,v in net.named_parameters():
            flag = True
            for out_layer in remove_list:
                if out_layer in k:
                    flag = False
                    break

            if (k in pre_net.keys()) and flag:
                v.data = pre_net[k].data
                print(k+" pre trained ok")

            else:
                if k.find("weight") >= 0:
                    nn.init.xavier_normal_(v.data)  # 没有预训练，则使用xavier初始化
                else:
                    nn.init.constant_(v.data, 0)  # bias 初始化为0
        print("load part of pre-train model {}...".format(pre_model))


def log_txt(filename, contexts=None, mark=False):
    f = open(filename, "a")
    if mark:
        sig = "------------------------------------------------\n"
        f.write(sig)
    else:
        f.write(contexts)
    f.close()

import os
def adjust_learning_rate(optimizer, epoch,lr,lr_change_rate,lr_change_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 steps"""
    lr = lr * (lr_change_rate ** int(epoch/lr_change_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr