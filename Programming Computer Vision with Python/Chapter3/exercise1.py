from PIL import Image
from numpy import *
from pylab import *
from scipy import ndimage

im = array(Image.open('empire.jpg').convert('L'))
H = array([[1.4,0.05,-100],[0.05,1.5,-100],[0,0,1]])
im2 = ndimage.affine_transform(im, H[:2,:2], (H[0,2],H[1,2]))

figure()
gray()
imshow(im2)
axis('off')
show()