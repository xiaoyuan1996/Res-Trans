import os
import mytools
import numpy as np
import random

root_path = "../data/image_data/train_cut/"


paths = os.listdir(root_path)

for path in paths:
    sub_path = root_path + path +"/"
    for file in os.listdir(sub_path):

        if random.uniform(0,1)>0.1:
            item = "_ | " + "train/" + path + "/" + file + " | " + path + "\n"
            mytools.log_to_txt(item,"trainB.txt")
        else:
            item = "_ | " + "val/" + file + " | " + file[9] + "\n"
            mytools.log_to_txt(item, "valB.txt")



