from PIL import Image
from numpy import *
from pylab import *

import warp
im1 = array(Image.open('smile.jpg').convert('L'))
im2 = array(Image.open('bg.jpg').convert('L'))

tp = array([[100,256,128,666],[300,283,567,233],[1,1,1,1]])

im3 = warp.image_in_image(im1, im2, tp)

figure()
gray()
imshow(im3)
axis('equal')
axis('off')
show()