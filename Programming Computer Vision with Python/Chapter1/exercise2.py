from PIL import Image
import sys, os
filelist = ['empire.jpg','empire2.jpeg']

for infile in filelist:
    outfile = os.path.splitext(infile)[0] + '.png'
    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print("cannot convert!")