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

def build_vocab(file_name):
    vocab = defaultdict(float)
    with open(file_name, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            orig_rev = clean_str(" ".join(rev))
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1

    return vocab

def load_vec(fname, vocab, layer_size):
    """
    Loads kx1 word vecs
    """
    if fname == "docNNSE300.txt":
        seperator = "\t"
    else:
        seperator = " "

    word_vecs = {}
    with open(fname, "rb") as f:
        binary_len = np.dtype('float32').itemsize * layer_size
        lines = f.readlines()
        vocab_size = len(lines)
        for line in lines:
            word = line.split(seperator)[0]
            if word in vocab:
               word_vecs[word] = np.fromstring(line[len(word):], dtype='float32', sep = seperator)
        
        print "vocabulary: " + str(vocab_size)

    return word_vecs

def get_vector(word_vecs, file_name, output_name, layer_size):
    vocab_size = len(word_vecs)
    with open(file_name, "rb") as f:
        output = codecs.open(file_name+"_"+output_name+"_vector.txt","w",encoding="utf-8")
        for line in f:
            vector = [] 
            rev = []
            cnt = 0
            rev.append(line.strip())
            orig_rev = clean_str(" ".join(rev))

            for word in orig_rev:
                if word in word_vecs:
                    cnt += 1
                    vector += word_vecs[word].tolist()
                if cnt == 50: break
            
            if len(vector) < 50*layer_size: vector += [0]*(50*layer_size - len(vector))
            vector = ["{:.5f}".format(num) for num in vector]
            output.write(" ".join(vector)+"\n")
        output.close()

if __name__=="__main__":
    sentence_file = sys.argv[1]
    vector_file = sys.argv[2]
    layer_size = int(sys.argv[3])
    output_name = sys.argv[4]
    print "sentence file name: " + sentence_file
    print "vector file name: " + vector_file
    print "size of layer: " + str(layer_size)
    print "ideal output name: " + output_name
    print "loading data...",        
    vocab = build_vocab(sentence_file)
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "loading vectors..."
    gv = load_vec(vector_file, vocab, layer_size)
    print "pre-trained vector loaded!"
    print "num words already in pre-trained vector: " + str(len(gv))
    get_vector(gv, sentence_file, output_name, layer_size)
    print "transform completed!"
