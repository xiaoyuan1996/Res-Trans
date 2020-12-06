import os
import mytools
import numpy as np
import random

root_path = "../../data/train/"


paths = os.listdir(root_path)

items = []
for path in paths:
    sub_path = root_path + path +"/"
    for file in os.listdir(sub_path):
        item = "_ | "+"train/"+path+"/"+file+" | "+path+"\n"
        items.append(item)

mytools.log_to_txt(items,"trainB.txt")


root_path = "../../../data/val/"

items = []
for file in os.listdir(root_path):
    item = "_ | "+"val/"+file+" | "+file[9]+"\n"
    items.append(item)

mytools.log_to_txt(items,"valB.txt")

