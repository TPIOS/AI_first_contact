import re
import sys


def main(dict_file, text_file, word_file):
    dict_file = open(dict_file,'r')
    text_file = open(text_file,'r')
    word_file = open(word_file,'w')

    lines = dict_file.readlines()
    big_dict = dict()

    for line in lines:
        idx, word, _ = re.split(r"\s|\n",line)
        big_dict[word] = idx

    lines = text_file.readlines()
    for line in lines:
        writer = []
        rec = ""
        for c in line:
            if 'a'<=c<='z':
                rec += c
            else:
                if rec in big_dict: 
                    writer.append(big_dict[rec])
                elif rec.isalpha(): 
                    writer.append('-1')
                if c == "\n":
                    writer.append(c)
                rec = ""
                
        word_file.write(" ".join(writer))
    
    dict_file.close()
    text_file.close()
    word_file.close()

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])