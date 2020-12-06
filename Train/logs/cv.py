import numpy as np
import matplotlib.pyplot as plt


plt.close()
plt.ion()
plt.grid()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2 = ax1.twinx()
fig.legend()

ii = 0
while True:
    info = np.load("logs.npy", allow_pickle=True)

    info = np.transpose(info)
    print(info)
    # epoch time loss acc
    epoch = info[0]
    time = info[1]
    loss = info[2]
    acc = info[3]

    print("Model has been trained {} epochs.".format(len(acc)))

    # 绘制

    #生成镜面坐标轴

    ax1.plot(epoch, loss, linewidth=1.0, linestyle='--', label='Loss',c='blue')
    ax2.plot(epoch, acc, linewidth=1.0, linestyle='-.', label='Acc', c='coral')
    ax2.set_ylabel('Acc')
    plt.grid()


    plt.title("Xunfei time:"+str(int(sum(time)/60))+"min/"+str(len(acc))+"epoch "
                    "\n ps:resnet-incep , bs=8, lr1.2e-4 "
                    "\n")
    if ii==0:
        fig.legend()
        ii += 1

    plt.pause(15)
    # plt.show()

