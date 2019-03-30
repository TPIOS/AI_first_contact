from PIL import Image

pil_im_1 = Image.open("empire.jpg")

pil_im_1.show()

box = (100,100,400,400)
region = pil_im_1.crop(box)
region = region.transpose(Image.ROTATE_180)

pil_im_1.paste(region,box)
pil_im_1.show()