import os
import cv2

root_path = ""
syss = ["../data/image_data/test/","../data/image_data/train/1/","../data/image_data/train/2/","../data/image_data/train/3/","../data/image_data/train/4/"]
# syss = ["test/"]

for path in syss:
    for i,img_path in enumerate(os.listdir(root_path+path)):
        if ".jpg" in img_path:
            if "enh_" in img_path:
                print(path," ",i,"/",len(os.listdir(root_path+path)))
                img = cv2.imread(root_path+path+img_path)
                dst = img[58:428,80:573]
                # cv2.imshow('image',dst)
                # cv2.waitKey(0)
                paths = root_path+path.replace("test/","test_cut/").replace("train","train_cut")+img_path
                # print(path)
                cv2.imwrite(
                    paths,
                    dst
                )