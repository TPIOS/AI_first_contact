from numpy import *
from PIL import Image
import sys, os
def get_imlist(path):
    ##返回目录中所有JPG图像的文件名列表
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.jpeg')]

def imresize(im, sz):
    ##使用PIL对象重新定义图像数组的大小
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))

def histeq(im, nbr_bins=256):
    ##对一副灰度图像进行直方图均衡化

    imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255*cdf / cdf[-1]

    im2 = interp(im.flatten(),bins[:-1],cdf)

    return im2.reshape(im.shape), cdf

def compute_average(imlist):
    ##计算图像列表的平均值
    averageim = array(Image.open(imlist[0]),'f')
    skip_im_count = 0
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            skip_im_count += 1
            print(imname + '...skipped')
    averageim /= len(imlist-skip_im_count)

    return array(averageim, 'uint8')
