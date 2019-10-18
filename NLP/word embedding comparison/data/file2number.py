import re
def process(file):
    index_file = open("enwiki2018_c50k.index","r")
    file = open(file,"r", encoding="utf-8", errors="replace")

    lines = index_file.readlines()
    big_dict = dict()

    for line in lines:
        idx, word, _ = re.split(r"\s|\n",line)
        big_dict[word] = idx

    lines = file.readlines()[2:]
    writer = []

    for line in lines:    
        rec = ""
        lower_line = line.lower()

        for c in lower_line:
            if 'a'<=c<='z':
                rec += c
            else:
                if rec in big_dict: 
                    writer.append(big_dict[rec])
                rec = ""
    
    index_file.close()
    file.close()

    return " ".join(writer)