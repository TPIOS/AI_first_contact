from PIL import Image
origin = Image.open("./images/4.jpg")
print(origin.size)
new = origin.resize((1280, 720))
new.show()

new.save("./images/4_inv.jpg")