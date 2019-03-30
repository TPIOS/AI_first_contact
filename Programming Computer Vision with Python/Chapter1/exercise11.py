from PIL import Image
from numpy import *
from scipy.ndimage import filters

im = array(Image.open('empire.jpg').convert('L'))
sigma = 5

imx = zeros(im.shape)
im1x = filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)

pil_im_1 = Image.fromarray(uint8(im1x))

imy = zeros(im.shape)
#pil_im_2 = Image.fromarray(uint8(filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)))

pil_im_1.show()
#pil_im_2.show()
