from PIL import Image

p = Image.open("images/school.jpg")
res = p.resize((400, 300))
res.save("images/school_res.jpg")