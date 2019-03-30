from PIL import Image
from numpy import *
from scipy.ndimage import filters

im = array(Image.open('empire.jpg').convert('L'))
im2 = filters.gaussian_filter(im, 10)

pil_im_1 = Image.fromarray(uint8(im))
pil_im_2 = Image.fromarray(uint8(im2))

pil_im_1.show()
pil_im_2.show()