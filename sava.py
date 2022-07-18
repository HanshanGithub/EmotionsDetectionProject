import numpy as np
from matplotlib import pyplot as plt

def pltFigure():
    img = plt.imread(r'E:\Python\Emotional\Emotions\img\cutA.png')
    plt.figure()
    plt.imshow(img)
    index=101
    plt.savefig(r'E:\Python\Emotional\Emotions\index\cut{0}.png'.format(index),dpi=300)
    plt.show()

if __name__ == '__main__':
    pltFigure()