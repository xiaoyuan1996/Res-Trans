# 路径定义
train_path = "utilize/trainB.txt"
val_path = "utilize/valB.txt"
test_path = "utilize/testB.txt"

model_path = "models/"
log_path = "logs/"

pre_model = "models/best.ckpt"

# 模型训练设置
pre_train = True

lr = 0.0005
# lr = 0.0012

lr_change_epoch = 10
lr_change_rate = 0.7

batchsize = 12
num_workers = 0
log_step = 10
save_epoch = 1
eval_epoch = 1
shuffle = True
num_epochs = 60

# 模型设置
QueryLSTM_embedsize=64
QueryLSTM_hiddensize=256
QueryLSTM_numlayers=2

# fine tune params
dropout_resnet = 0.2
pooling_size_resnet = 3
dropout_linear = 0.4
resnet_layers = [4,4,2,1]
crop_size = 278

init_xavri = "True"
SGD = "False"


