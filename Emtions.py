import os

import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QAction, QMessageBox, QInputDialog
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

import get_data
import kmeans
import knn
import ui_emtion
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

class emotionWindow(QtWidgets.QMainWindow):
    clicktime = 0

    knndata = []
    knnlable = []
    knnPreLable = []

    kmeansdata = []
    kmeanslable = []
    kmeansPreLable = []
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = ui_emtion.Ui_Emotions()
        self.ui.setupUi(self)
        # 给button 的 点击动作绑定一个事件处理函数
        # self.ui.knn.clicked()
        self.ui.cutA.triggered.connect(self.showA)
        self.ui.cutB.triggered.connect(self.showB)
        self.ui.id.triggered.connect(self.myabout)
        self.ui.interface1.triggered.connect(self.Kmeans)
        self.ui.interface2.triggered.connect(self.KNN)
        self.ui.index1.triggered.connect(self.indexKmeans)
        self.ui.index2.triggered.connect(self.indexKnn)
        self.ui.start1.triggered.connect(self.startKmeans)
        self.ui.start2.triggered.connect(self.startKnn)

    def pics(self,path):
        if os.path.exists(path) == False:
            self.ui.text.append("   error!\n")
        frame = QImage(path)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)  # fitInView(item)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.ui.pic.setScene(scene)
        self.ui.pic.fitInView(item)
        self.ui.pic.show()

    def showA(self):
        self.clicktime += 1
        self.ui.text.append("step{0}:原始数据".format(self.clicktime))
        path = r'E:\Python\Emotional\Emotions\img\cutA.png'
        self.pics(path)

    def showB(self):
        self.clicktime += 1
        self.ui.text.append("step{0}:人脸对齐".format(self.clicktime))
        path = r'E:\Python\Emotional\Emotions\img\cutB.png'
        self.pics(path)

    def myabout(self):
        reply = QMessageBox.about(self, "软件说明", "Demo:2.3.1\n3120190912213\n黄天鑫\n2022.06.01")

    def Kmeans(self):
        self.clicktime += 1
        self.ui.text.append("step{0}:K-Means聚类效果".format(self.clicktime))
        path = r'E:\Python\Emotional\Emotions\img\k-means.png'
        self.pics(path)

    def KNN(self):
        self.clicktime += 1
        self.ui.text.append("step{0}:K-Means聚类效果".format(self.clicktime))
        path = r'E:\Python\Emotional\Emotions\img\knn.png'
        self.pics(path)

    def indexKmeans(self):
        self.clicktime += 1
        index, ok = QInputDialog.getInt(self, "查看聚类效果", "请输入位置(0-4000):",2213,0,4000)
        self.ui.text.append("step{0}:查看第{1}个聚类效果[P:{2},T:{3}]".
                            format(self.clicktime, index, self.kmeansPreLable[index], self.kmeanslable[index]))
        if self.kmeansdata == []:
            reply = QMessageBox.critical(self, '错误', '未进行K-Means聚类', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        self.drawIndex('kmeans',index)

    def indexKnn(self):
        self.clicktime += 1
        index, ok = QInputDialog.getInt(self, "查看聚类效果", "请输入位置(0-600):", 213, 0, 600)
        self.ui.text.append("step{0}:查看第{1}个分类效果[P:{2},T:{3}]".
                            format(self.clicktime,index,self.knnPreLable[index], self.knnlable[index]))
        if self.knndata == []:
            reply = QMessageBox.critical(self, '错误', '未进行K-Means聚类', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        self.drawIndex('knn',index)

    def startKmeans(self):
        self.clicktime += 1
        self.ui.text.append("step{0}:进行K-Means聚类".format(self.clicktime))
        try:
            t1 = get_data.Train()
            self.kmeansdata, self.kmeanslable = t1.next_batch(4000)
            s = KMeans(init='random', n_clusters=7).fit(self.kmeansdata)
            self.kmeansPreLable = s.labels_ # 每个样本的类别
            kmeans.anasys(self.kmeansdata,self.kmeanslable,self.kmeanslable)
            self.ui.text.append("   K-Means聚类完毕！")
        except:
            self.ui.text.append("   K-Means出错")

    def startKnn(self):
        try:
            self.clicktime += 1
            self.ui.text.append("step{0}:进行K-Means聚类".format(self.clicktime))
            E1 = get_data.Eval()
            data, label = E1.get_data()
            np.random.shuffle(data)
            np.random.shuffle(label)

            data_train = data[:int(data.shape[0] * 0.8)]
            label_train = label[:int(label.shape[0] * 0.8)]

            self.knndata = data[int(data.shape[0] * 0.8):]
            self.knnlable = label[int(label.shape[0] * 0.8):]

            # 训练模型
            clf = KNeighborsClassifier()
            clf.fit(data_train, label_train)
            self.knnPreLable = clf.predict(self.knndata)
            knn.knnAnaysis(self.knndata,self.knnlable,self.knnPreLable)
            self.ui.text.append("   KNN分类完毕！")
        except:
            self.ui.text.append("   KNN出错")

    def drawIndex(self,type,index):
        if type == 'kmeans':
            data1 = self.kmeansdata[index]
        if type == 'knn':
            data1 = self.knndata[index]
        img = []
        for i in data1:
            j = int(i)
            img.append(j)
        IMG = np.mat(img)
        plt.figure()
        plt.imshow(IMG.reshape(48, 48), cmap='gray')
        if type == 'kmeans':
            plt.title('Predict:{0},Ture:{1}'.format(self.kmeansPreLable[index], self.kmeanslable[index]))
            plt.savefig(r'E:\Python\Emotional\Emotions\index\kmeans{0}.jpg'.format(index), dpi=300)
            path = r'E:\Python\Emotional\Emotions\index\kmeans{0}.jpg'.format(index)
        if type == 'knn':
            plt.title('Predict:{0},Ture:{1}'.format(self.knnPreLable[index], self.knnlable[index]))
            plt.savefig(r'E:\Python\Emotional\Emotions\index\knn{0}.jpg'.format(index), dpi=300)
            path = r'E:\Python\Emotional\Emotions\index\knn{0}.jpg'.format(index)

        self.pics(path)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = emotionWindow()
    window.show()
    sys.exit(app.exec_())