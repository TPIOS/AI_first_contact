import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8') 

def process(file_name):
    file = open(file_name,"r", encoding="utf-8", errors="replace")
    lines = file.readlines()[2:]

    writer = []
    for line in lines:
        rec = ""
        lower_line = line.lower()
        for c in lower_line:
            if c != "\n" and c != " ":
                rec += c
            else:
                if rec != "": writer.append(rec)
                rec = ""
    
    return " ".join(writer)

folder_path = "./20_newsgroup/"

for folder in os.listdir(folder_path):
    folder_name = folder_path+folder

    res_file = open(folder_name+".oneline",'w', encoding="utf-8")
    res_write = ""

    file_path = folder_name+"/"
    for file in os.listdir(file_path):
        file_name = file_path + file
        res_write += process(file_name) + '\n'

    res_file.write(res_write)
    res_file.close()