import numpy as np
import pandas as pd
from collections import defaultdict
import sys, re
import codecs

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def build_vocab(data_folder):
    vocab = defaultdict(float)
    for filename in data_folder:
        with open(filename, "rb") as f:
            for line in f:       
                rev = []
                rev.append(line.strip())
                orig_rev = clean_str(" ".join(rev))[2:]
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1

    return vocab

def load_vec(fname, vocab):
    """
    Loads kx1 word vecs
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        headline = f.readline()
        vocab_size, layer_size = map(int, headline.split(" "))
        binary_len = np.dtype('float32').itemsize * layer_size
        lines = f.readlines()
        for line in lines:
            word = line.split(" ")[0]
            if word in vocab:
               word_vecs[word] = np.fromstring(line[len(word):], dtype='float32', sep = " ")
        
        print "vocabulary: " + str(vocab_size)

    return word_vecs

def get_vector(word_vecs, data_folder, k=52):
    for filename in data_folder:
        with open(filename, "rb") as f:
            output = codecs.open(filename+"_"+str(k)+"vector.txt","w",encoding="utf-8")
            for line in f:
                vector = [] 
                rev = []
                cnt = 0
                rev.append(line.strip())
                orig_rev = clean_str(" ".join(rev))[2:]

                for word in orig_rev:
                    if word in word_vecs:
                        cnt += 1
                        vector += ["{:.5f}".format(num) for num in word_vecs[word].tolist()]
                    if cnt == 50: break
                
                if len(vector) < 50*k: vector += ["0"]*(50*k - len(vector))
                output.write(" ".join(vector)+"\n")
            output.close()

if __name__=="__main__":    
    vector_file = sys.argv[1]     
    data_folder = ["stsa.binary.test","stsa.binary.train"]    
    print "loading data...",        
    vocab = build_vocab(data_folder)
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "loading vectors..."
    gv = load_vec(vector_file, vocab)
    print "pre-trained vector loaded!"
    print "num words already in pre-trained vector: " + str(len(gv))
    get_vector(gv, data_folder)
    print "transform completed!"