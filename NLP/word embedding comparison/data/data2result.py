import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8') 
from file2number import process

folder_path = "./20_newsgroup/"

for folder in os.listdir(folder_path):
    folder_name = folder_path+folder

    res_file = open(folder_name+".number",'w', encoding="utf-8")
    res_write = ""

    file_path = folder_name+"/"
    for file in os.listdir(file_path):
        file_name = file_path + file
        res_write += process(file_name) + '\n'

    res_file.write(res_write)
    res_file.close()