import mytools
import os
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm


root = "allRes/"
ensemble_file = os.listdir(root)

# 获取整合结果
res = {}
for file in ensemble_file:
    ctx = mytools.load_from_json(root+file)
    for k, v in ctx.items():
        k = int(k.replace("test/","").replace("enh_","").replace(".jpg","").replace("_w",""))
        if k in res.keys():
            # res[k] += v
            res[k] = [res[k][i] + v[i] for i in range(0, len(v))]
        else:
            res[k] = v

#--------------------------------------------------------------------
# 产生csv
result = []
for k in sorted(res):
    cls = np.argmax(res[k])+1
    result.append([k,cls])
data = pd.DataFrame(result, index=None, columns=["id", "label"])
data.to_csv("submission.csv", index=None)
print("Saved finished.")




