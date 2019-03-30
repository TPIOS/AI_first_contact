from PIL import Image

pil_im_1 = Image.open("empire.jpg")

pil_im_1.show()

pil_im_1.thumbnail((128,128))
pil_im_1.show()


##pil_im_2 = Image.open("empire.jpg").convert('L')