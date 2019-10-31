import os
import math
import sys
import datetime
from copy import deepcopy
import numpy as np
import pickle
import re, string

def loadModel(model_file):
    modelFile = open(model_file, "rb")
    data = pickle.load(modelFile)
    table = data['table']
    vocab = data['vocab']
    tags = data['tags']
    tagAndWord = data['tagAndWord']
    modelFile.close()
    return table, vocab, tags, tagAndWord

def is_number(num):
    if num == "'" or num == "'s": return False
    pattern = re.compile(r'^[-+]?[\']?([0-9]+[.,-/]?)*[s]?$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False

def processWord(unkProb, x, wordIdx, tag, tagAndWord):
    # if is_number(x) or (x == "hundred" or x == "thousand" or x == "TWO" or x == "Sept.30" or x.endswith("1st") or x.endswith("2nd") or x.endswith("3rd")): #for all number, it should be number
    #     if tag == "CD":
    #         return unkProb * 10000

    # if x.endswith("-year-old"):
    #     if tag == "JJ":
    #         return unkProb * 10000

    #end-with-ing
    if x.endswith("ing"):
        if tag == "JJ":
            return unkProb * 5000
        elif tag == "NN":
            return unkProb * 5000
        elif tag == "VBG":
            return unkProb * 5000

    #end-with-ment/ness
    if x.endswith("ment") or x.endswith("ness") or x.endswith("tion") or x.endswith("sion"):
        if tag == "NN":
            return unkProb * 10000
        
    #capitalized word
    if x == x.capitalize() and wordIdx != 0:
        if x.endswith("es"):
            if tag == "NNPS":
                return unkProb * 7500
            elif tag == "JJ":
                return unkProb * 5000
        else:
            if tag == "NNP":
                return unkProb * 7500
            elif tag == "JJ":
                return unkProb * 5000

    #abbr.
    if x.isupper() and wordIdx != 0:
        if tag == "NNP" or tag == "NNPS":
            return unkProb*10000

    #dash
    if "-" in x:
        if x.endswith("s"):
            if tag == "NNPS":
                return unkProb * 7500
            elif tag == "NNS" or tag == "JJ":
                return unkProb * 5000
        else:
            if tag == "NNP":
                return unkProb * 7500
            elif tag == "NNS" or tag == "JJ":
                return unkProb * 5000

    #end-with-s
    if x.endswith("s"):
        if x[:-1] in tagAndWord["NN"]:
            if tag == "NNS":
                return unkProb * 10000
        elif x[:-1] in tagAndWord["NNP"]:
            if tag == "NNPS":
                return unkProb * 10000
        elif x[:-1] in tagAndWord["VB"] or x[:-1] in tagAndWord["VBP"]:
            if tag == "VBZ":
                return unkProb * 10000

    if x+"s" in tagAndWord["NNS"]:
        if tag == "NN":
            return unkProb * 10000
    
    if x+"s" in tagAndWord["VBZ"]:
        if tag == "VB" or tag == "VBP":
            return unkProb * 10000
    
    return unkProb

def algorithm(words, table, vocab, tags, tagAndWord):
    viterbi = list()
    cntTags = len(tags)
    prevCol = [(0, 0) for i in range(cntTags)] #second 0 means <s>
    for wordIdx, word in enumerate(words):
        # word = processWord(word, vocab, tags, tagAndWord)
        nextCol = list()
        if word in vocab:
            for tagIdx, tag in enumerate(tags):
                if word in tagAndWord[tag]:
                    log_prob = math.log2(tagAndWord[tag][word])
                else:
                    log_prob = -999
                log_list = list()
                for tempIdx, preTag in enumerate(prevCol):
                    prevIdx = tempIdx + 1 if wordIdx != 0 else 0
                    log_t = math.log2(table[prevIdx][tagIdx])
                    log_list.append(log_t + preTag[0])
                max_log_prob = max(log_list)
                max_log_prob_pair = (max_log_prob + log_prob, log_list.index(max_log_prob))
                nextCol.append(max_log_prob_pair)
        else:
            for tagIdx, tag in enumerate(tags):
                unkProb = tagAndWord[tag]["<UNK>"]
                unkProb = processWord(unkProb, word, wordIdx, tag, tagAndWord)
                if unkProb == 0:
                    log_prob = -999
                else:
                    log_prob = math.log2(unkProb)
                log_list = list()
                for tempIdx, preTag in enumerate(prevCol):
                    prevIdx = tempIdx + 1 if wordIdx != 0 else 0
                    log_t = math.log2(table[prevIdx][tagIdx])
                    log_list.append(log_t + preTag[0])
                max_log_prob = max(log_list)
                max_log_prob_pair = (max_log_prob + log_prob, log_list.index(max_log_prob))
                nextCol.append(max_log_prob_pair)
        viterbi.append(deepcopy(nextCol))
        prevCol = deepcopy(nextCol)

    lastCol = viterbi[len(viterbi) - 1]
    lastRow = [(lastCol[i][0] + math.log2(table[i][cntTags]), i) for i in range(cntTags)] 
    lastIdx = max(lastRow, key = lambda x: x[0])[1]
    ansTagsIdx = [lastIdx]
    for col in viterbi[::-1]:
        ansTagsIdx.append(col[lastIdx][1])
        lastIdx = col[lastIdx][1]
    
    ansTagsIdx = ansTagsIdx[::-1]
    return [tags[i] for i in ansTagsIdx[1:]]# the 0th predicts for <s>
                
def tag_sentence(test_file, model_file, out_file):
    table, vocab, tags, tagAndWord = loadModel(model_file)
    testFile = open(test_file, "r")
    outFile = open(out_file, "w")
    lines = testFile.readlines()
    for line in lines:
        sentence = line.rstrip()
        if sentence[-1] in string.punctuation or sentence.endswith("Baltimore") or sentence.endswith("Treasury Securities") or sentence.endswith("Mortgage-Backed Issues") or sentence.endswith("Foreign Bond"):
            pass
        else:
            sentence = sentence.lower()

        resTags = algorithm(sentence.split(" "), table, vocab, tags, tagAndWord)

        sentence = line.rstrip()
        words = sentence.split(" ")
        ans = list()
        for i in range(len(resTags)): 
            x = words[i]
            if is_number(x) or (x == "hundred" or x == "thousand" or x == "TWO" or x == "Sept.30" or x.endswith("1st") or x.endswith("2nd") or x.endswith("3rd")):
                resTags[i] = "CD"
            if words[i].endswith("-year-old"):
                resTags[i] = "JJ"
            ans.append(words[i]+"/"+resTags[i])
        ans = " ".join(ans)
        outFile.write(ans+"\n")

    print('Finished...')
    outFile.close() 
    testFile.close() 

if __name__ == "__main__":
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)