import numpy as np

import get_data
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt



def kmeans(data):
    s = KMeans(init='random', n_clusters=7).fit(data)
    print(s)
    label_predict = s.labels_

    # print clf.cluster_centers_
    # 每个样本所属的簇
    print(s.labels_)
    # 用来评估簇的个数是否合适距离越小说明簇分的越好
    print("inertia is ", s.inertia_)
    return label_predict

def anasys(data,label_predict,label):
    x = range(data.shape[0])
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_title('7 of Predict cluster')
    ax2.set_title('True cluster')
    plt.xlabel('samples')
    plt.ylabel('label')
    ax1.scatter(x, label_predict, c=label_predict, marker='o')
    ax2.scatter(x, label, c=label, marker='o')
    # print('len:',len(x))
    plt.savefig(r'E:\Python\Emotional\Emotions\img\k-means.png', dpi=300)
    # plt.show()

def draw48(data):
    plt.figure()
    # img = data.reshape(48,48)
    plt.imshow(data,cmap='gray')
    plt.show()

def start_kmeans():
    t1 = get_data.Train()
    randx1, randy1 = t1.next_batch(4000)
    index = 2711
    label_predict = kmeans(randx1)
    accNum = 0

    for i in range(len(randy1)):
        if(randy1[i]==label_predict[i]):
            accNum += 1
    print('acc:',accNum/len(randy1))
    anasys(randx1, label_predict, randy1)
    data1 = randx1[index]
    img = []
    for i in data1:
        j = int(i)
        img.append(j)
    IMG = np.mat(img)
    # print(IMG)
    draw48(IMG.reshape(48, 48))
    print('Predict emotion:', label_predict[index])
    print('Ture emotion:', randy1[index])

if __name__ == '__main__':
    start_kmeans()