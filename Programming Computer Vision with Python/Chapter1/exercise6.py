from PIL import Image
from numpy import *
im = array(Image.open('empire.jpg').convert('L'))
pil_im_1 = Image.fromarray(im)
pil_im_1.show()

im2 = 255 - im
pil_im_2 = Image.fromarray(uint8(im2))
pil_im_2.show()

im3 = (100.0/255) * im + 100
pil_im_3 = Image.fromarray(uint8(im3))
pil_im_3.show()

im4 = 255.0 * (im/255.0)**2
pil_im_4 = Image.fromarray(uint8(im4))
pil_im_4.show()