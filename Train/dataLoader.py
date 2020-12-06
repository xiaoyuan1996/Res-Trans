import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from PIL import Image
import config
import mytools
import random

def resort(af):
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    len_af = np.shape(af)[1]

    for i in range(len(af)):
        # af[i] = normalization(af[i])*255
        af[i] = normalization(af[i])

    # af = np.resize(af, (367,489)).astype(int)

    bf = af*0+ int(255*(len_af/1500))

    cf = np.stack((af,bf), axis=2)
    cf = Image.fromarray(np.uint8(cf))

    return af

def downSample(fbank, rate):
    downSample_fbank = []
    i = 0
    frame_bank = 0
    for bank in fbank:
        frame_bank += bank
        if i%rate ==0:
            downSample_fbank.append(frame_bank/rate)
            frame_bank = 0
        i+=1

    return np.array(downSample_fbank)

def cut_out(vf, cut_size = 50,  padding_number=3):
    size_length, size_width = vf.shape[1],vf.shape[2]

    cut = torch.ones(3,size_length,size_width)

    # 建立pad
    cut_pad = torch.ones(3,cut_size,cut_size)

    # 选随机值 叠加
    for i in range(padding_number):
        rand_pos_x = random.randint(0,int(size_length/padding_number)-cut_size-1)
        rand_pos_y = random.randint(0,int(size_width/padding_number)-cut_size-1)
        cut[:, rand_pos_x:rand_pos_x+cut_size, rand_pos_y:rand_pos_y+cut_size] -= cut_pad
    vf = torch.mul(vf, cut)

    # rand_pos_x = random.randint(0,size_length-50-1)
    # rand_pos_y = random.randint(0,size_width-50-1)
    # cut[:, rand_pos_x:rand_pos_x+50, rand_pos_y:rand_pos_y+50] -= cut_pad

    # 相乘
    # vf = torch.mul(vf, cut)

    return vf


def calc_af_feature(af ):
    af = af.mean(axis=1)
    af.astype(np.float)
    return af

class AudioDataset(data.Dataset):
    def __init__(self,  train="train", transform=None):
        print("Reading data...")

        self.img_path = "../data/argumented_data/"

        if train =="train":
            data_path = config.train_path
        elif train == "val":
            data_path = config.val_path
        else:
            data_path = config.test_path

        #读取数据
        self.data = mytools.load_from_txt(data_path)

        self.transform = transform

        self.train = train


    def __getitem__(self, index):
        item = self.data[index]

        _, data_path, label = item.replace("\n","").split(" | ")

        try: # test时使用
            label = int(label)-1
        except:
            label = None

        # 图片
        if "_cut" not in data_path:
            # vf = self.img_path+ data_path.replace("test/","test_cut/enh_").replace("train/1/","train_cut/1/enh_")\
            #     .replace("train/2/","train_cut/2/enh_").replace("train/3/","train_cut/3/enh_").replace("train/4/","train_cut/4/enh_")
            vf = self.img_path + data_path
        else:
            vf = "../data/"+data_path[3:]

        # if self.train == "train":
        #     singer = int(vf[38:40])-36
        # else:
        #     singer = -1


        # if self.train == "train":
        #     # 虚假图片
        #     vf_noise = random.choice(self.data)
        #     # print(vf_noise)
        #     _, noise_path, _ = vf_noise.replace("\n","").split(" | ")
        #     # 图片
        #     if "_cut" not in noise_path:
        #         vf_noise = self.img_path+ noise_path.replace("test/","test_cut/enh_").replace("train/1/","train_cut/1/enh_")\
        #             .replace("train/2/","train_cut/2/enh_").replace("train/3/","train_cut/3/enh_").replace("train/4/","train_cut/4/enh_")
        #     else:
        #         vf_noise = "../data/"+noise_path[3:]
        #
        #     img_1 = cv2.imread(vf)
        #     img_2 = cv2.imread(vf_noise)
        #     weight = np.float(random.randint(55, 95) / 100.0)
        #     dst = cv2.addWeighted(img_1, weight, img_2, 1 - weight, 0)
        #     cv2.imwrite("utilize/tmp.jpg", dst)
        #
        #     # af = "../data/audio_feature/" + data_path.replace("jpg","npy")
        #     # af = mytools.load_from_npy(af)
        #     #
        #     # af = calc_af_feature(af)
        #     # af = torch.FloatTensor(af)
        #     vf = "utilize/tmp.jpg"

        vf = Image.open(vf).convert('RGB')



        if self.transform is not None:
            vf = self.transform(vf)

        if self.train == "train":
            vf = cut_out(vf)
        # if self.transform_cf is not None:
        #     cf = self.transform_cf(cf)
        # vf = torch.cat([vf,cf], dim=0)

        # 读出名称
        name = data_path.replace("../test/","")

        return vf,label,name
        # return vf,label,name,singer

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    # 按照文本个数排序
    vf,labels,name = zip(*data)
    # af,vf,labels,name = zip(*data)


    # af_t = torch.stack(af,0)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    vf_t = torch.stack(vf, 0)

    # Merge label
    label = []
    for i in labels:
        label.append([i])
    label = torch.LongTensor(label).view(1, len(labels))[0]

    # # Merge singer
    # singer = []
    # for i in singers:
    #     singer.append([i])
    # singer = torch.LongTensor(singer).view(1, len(singers))[0]


    return vf_t, label,name
    # return af_t,vf_t, label,name



def get_loader(train="train",batchsize=config.batchsize, num_workers=config.num_workers):
    if train=="train":
        transform = transforms.Compose([
            transforms.Resize((config.crop_size, config.crop_size)),
            # transforms.RandomCrop(256),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
            # transforms.Normalize((0.506, 0.589, 0.496),
            #                      (0.237, 0.264, 0.235))])
    else:
        transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.Resize((256, 256)),
            # transforms.RandomCrop(256),
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

    # if train=="train":
    #     transform = transforms.Compose([
    #         transforms.Resize((278, 278)),
    #         transforms.RandomCrop(256),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406),
    #                              (0.229, 0.224, 0.225))])
    # else:
    #     transform = transforms.Compose([
    #         transforms.Resize((256, 256)),
    #         # transforms.CenterCrop(256),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406),
    #                              (0.229, 0.224, 0.225))])

    dataset = AudioDataset(train=train, transform=transform)
    if train=="train":
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            collate_fn= collate_fn,
            num_workers=num_workers,
            shuffle=True
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            collate_fn= collate_fn,
            num_workers=num_workers,
            shuffle=False
        )
    return data_loader

if __name__=='__main__':

    test_loader = get_loader(train="train")
    total_step = len(test_loader)

    import time
    # for i,(af_t,vf_t, label,name) in enumerate(test_loader):
    for i, ( vf_t, label, name) in enumerate(test_loader):

        print("-----------------")
        print(i)
        # print(af_t)
        # print(af_t.shape)
        print(vf_t.shape)
        # print(vf_t[0])
        print(label)
        # print(label.shape)
        # time.sleep(2)


