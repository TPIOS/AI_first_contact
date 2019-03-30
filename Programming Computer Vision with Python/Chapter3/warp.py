from PIL import Image
from numpy import *
from pylab import *
from scipy import ndimage
import homography

def image_in_image(im1,im2,tp):
    ##使用仿射变换将im1放置在im2上，使im1图像的角和tp尽可能的靠近。
    ##tp是齐次表示的，并且是按照从左上角逆时针计算的
    m,n = im1.shape[:2]
    fp = array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

    H = homography.Haffine_from_points(tp, fp)
    im1_t = ndimage.affine_transform(im1, H[:2,:2],(H[0,2], H[1,2]),im2.shape[:2])
    alpha = (im1_t > 0)

    return (1-alpha)*im2 + alpha*im1_t
