# -*- coding: utf-8 -*-  
import gensim 
import codecs    
  
def main():  
    path_to_model = 'word2vec.bin'  
    output_file = 'file.txt'  
    bin2txt(path_to_model, output_file)  
  
  
def bin2txt(path_to_model, output_file):  
    output = codecs.open(output_file, 'w' , 'utf-8')  
    model = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=True)  
    print('Done loading Word2Vec!')  
    vocab = model.vocab  
    for item in vocab:  
        vector = list()  
        for dimension in model[item]:  
            vector.append(str(dimension))  
        vector_str = ",".join(vector)  
        line = item + "\t"  + vector_str   
        output.writelines(line + "\n")
    output.close()  
  
if __name__ == "__main__":  
    main() 