import torch
import torch.nn as nn
import numpy as np
import random
# 设置随机数种子
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
# random.seed(seed)
np.random.seed(seed)
import os
from dataLoader import get_loader
import pandas as pd
import config
import time
from utilize import utilize
from tqdm import tqdm
from layers.resnet50 import model
from layers.capsnet import model
from layers.densenet import model
from layers.dpn import model
from layers.resnet_inception import model
from layers.resnext import model
from layers.transformer import model

# from layers.resnet3 import model

import random
import mytools

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
# net = model().to(device)
net = model().to(device)

# net = model().to(device)
utilize.load_pre_model(net, "models/best.ckpt")

# checkpoints = torch.load("models/cx_98.pth")
# net.load_state_dict(checkpoints)

# 加载测试集数据
test_loader = get_loader(train="test")
test_step = len(test_loader)

# 测试
print("Start test ...")
net.eval()

result,names = [],[]
save_to_json = []
for i, (vf, label,name) in enumerate(test_loader):
    # af = af.to(device)
    vf = vf.to(device)
    # out = net(vf)
    out, _ = net(vf)

    # print(out.data.cpu().numpy())

    # change weight
    save_to_json.extend(out.data.cpu().numpy())
    pred_y = torch.max(out, 1)[1].data.cpu().numpy()
    # print(pred_y)
    # print(list(name))
    result.extend(pred_y)
    names.extend(list(name))

# 转换为dict
query_dict = {}
for (n,r) in zip(names, result):
    print(n)
    query_dict[int(n.replace("test/","").replace(".jpg","").replace("enh_",""))] = r+1

# 转换为csv
result = []
for k in sorted(query_dict):
  result.append([k,query_dict[k]])
data = pd.DataFrame(result, index=None, columns=["id", "label"])
data.to_csv("submission.csv", index=None)


# 保存为子结果 后期做模型整合
import mytools
json_ = {}
for (n,j) in zip(names, save_to_json):
    json_[n] = [np.float(tmp) for tmp in j]
mytools.save_to_json(json_, "oneOfResult/only_af"+str(random.randint(0,1000))+".json")
