import mytools
import cv2
import random
import os
import numpy as np
from tqdm import tqdm

def make_noise_data(datasets, idx):
    item1 = random.choice(datasets).replace("\n", "")
    label = item1[-1]
    file_1 = "../data/image_data/train_cut/"+item1.replace("train","").split(" | ")[1]

    label_2 = label
    while label_2 == label:
        item2 = random.choice(datasets).replace("\n", "")
        label_2 = item2[-1]
    file_2 = "../data/image_data/train_cut/"+item2.replace("train","").split(" | ")[1]

    img_1 = cv2.imread(file_1)
    img_2 = cv2.imread(file_2)

    weight = np.float(random.randint(55,95)/100.0)
    dst = cv2.addWeighted(img_1, weight, img_2,1-weight ,0)

    cv2.imwrite("../data/argumented_data/train/"+label+"/"+str(idx)+".jpg", dst)

datasets = mytools.load_from_txt("../ExtractImage/trainB.txt")

for i in tqdm(range(4)):
    make_noise_data(datasets,i)

