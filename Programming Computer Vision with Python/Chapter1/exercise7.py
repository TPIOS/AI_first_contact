from PIL import Image
from numpy import *
import imtools

pil_im_0 = Image.open("hsy.JPG")
im = array(Image.open('hsy.JPG').convert('L'))
im2, cdf = imtools.histeq(im)

pil_im_1 = Image.fromarray(uint8(im))
pil_im_2 = Image.fromarray(uint8(im2))

pil_im_0.show()
pil_im_1.show()
pil_im_2.show()