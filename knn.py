import numpy as np
from matplotlib import pyplot as plt

import get_data
from sklearn.neighbors import KNeighborsClassifier



def knn(data,label):
    # 80%数据用于训练，20%数据用于测试
    data_train = data[:int(data.shape[0] * 0.8)]
    label_train = label[:int(label.shape[0] * 0.8)]
    data_test = data[int(data.shape[0] * 0.8):]
    label_test = label[int(label.shape[0] * 0.8):]
    # 训练模型
    clf = KNeighborsClassifier()
    clf.fit(data_train, label_train)
    label_predict = clf.predict(data_test)
    knnAnaysis(data_test, label_test, label_predict)
    print(clf.score(data_test,label_test))
    index = 2
    data1 = data[index]
    img = []
    for i in data1:
        j = int(i)
        img.append(j)
    IMG = np.mat(img)
    # print(IMG)
    draw48(IMG.reshape(48, 48))
    print('Predict emotion:', label_predict[index])
    print('Ture emotion:', label[index])

def knnAnaysis(data_test,label_test,label_predict):
    # 与真实标签比较
    x = range(data_test.shape[0])
    print(len(x))
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_title('Predict class')
    ax2.set_title('True class')
    plt.xlabel('samples')
    plt.ylabel('label')
    ax1.scatter(x, label_predict, c=label_predict, marker='o')
    ax2.scatter(x, label_test, c=label_test, marker='s')
    plt.savefig('E:\Python\Emotional\Emotions\img\knn.png',dpi=300)
    # plt.show()


def draw48(data):
    plt.figure()
    # img = data.reshape(48,48)
    plt.imshow(data,cmap='gray')
    plt.show()

def startKNN():
    E1 = get_data.Eval()
    data, label = E1.get_data()
    knn(data, label)

if __name__ == '__main__':
    startKNN()