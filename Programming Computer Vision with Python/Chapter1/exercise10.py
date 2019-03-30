from PIL import Image
from numpy import *
from scipy.ndimage import filters

im = array(Image.open('empire.jpg').convert('L'))

imx = zeros(im.shape)
filters.sobel(im,1,imx)


imy = zeros(im.shape)
filters.sobel(im,0,imy)


magnitude = sqrt(imx**2 + imy**2)
