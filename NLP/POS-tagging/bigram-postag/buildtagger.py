import os
import sys
import datetime
import numpy as np
from copy import deepcopy
import pickle

def saveModel(model_file, table, vocab, tags, tagAndWord):
    modelFile = open(model_file, "wb")
    data = {}
    data['table'] = table
    data['vocab'] = vocab
    data['tags'] = tags
    data['tagAndWord'] = tagAndWord
    pickle.dump(data, modelFile)
    modelFile.close()

def processWord(word, tag, prev):
    if (prev == "<s>" or prev == "``" or prev == "''" or prev == "-LRB-" or prev == ":") and  \
       (tag != "NNP" or tag != "NNPS"):
        word = word.lower()
    return word

def enlargeCorpus(vocab, tagAndWord, word, tag):
    if tag == "NN":
        if word+"s" in tagAndWord["NNS"]:
            tagAndWord["NNS"][word+'s'] += 1
        
    if tag == "NNP":
        if word+"s" in tagAndWord["NNPS"]:
            tagAndWord["NNPS"][word+'s'] += 1

    if len(word) >= 3:            
        if tag == "NNS":
            if word.lower().endswith("s") and word[:-1] in tagAndWord["NN"]: #s
                tagAndWord["NN"][word[:-1]] += 1
            elif word.lower().endswith("es") and word[:-2] in tagAndWord["NN"]: #es
                tagAndWord["NN"][word[:-2]] += 1 
            elif word.lower().endswith("ies") and word[:-3] + "y" in tagAndWord["NN"]: #ies
                tagAndWord["NN"][word[:-3] + "y"] += 1 

        if tag == "NNPS":
            if word.lower().endswith("s") and word[:-1] in tagAndWord["NNP"]: #s
                tagAndWord["NNP"][word[:-1]] += 1
            elif word.lower().endswith("es") and word[:-2] in tagAndWord["NNP"]: #es
                tagAndWord["NNP"][word[:-2]] += 1 
            elif word.lower().endswith("ies") and word[:-3] + "y" in tagAndWord["NNP"]: #ies
                tagAndWord["NNP"][word[:-3] + "y"] += 1 

        if tag == "NN":
            if word != word.lower():
                if word.lower() not in tagAndWord["NN"]:
                    tagAndWord["NN"][word.lower()] = 1
                    vocab.add(word.lower())
                else:
                    tagAndWord["NN"][word.lower()] += 1
        
        if tag == "NNS":
            if word != word.lower():
                if word.lower() not in tagAndWord["NNS"]:
                    tagAndWord["NNS"][word.lower()] = 1
                    vocab.add(word.lower())
                else:
                    tagAndWord["NNS"][word.lower()] += 1
        
        if tag == "NNP":
            if word.lower() in tagAndWord["NN"]:
                tagAndWord["NN"][word.lower()] += 1

def loadTrainData(train_file, vocab, tags, tagAndWord):
    corpus = dict()
    trainFile = open(train_file, "r")
    lines = trainFile.readlines()
    for line in lines:
        prev = "<s>"
        sentence = line.rstrip()
        sents = sentence.split(" ")
        for sent in sents:
            word, tag = sent.rsplit("/", 1)
            word = processWord(word, tag, prev)
            if tag not in tags:
                tags.append(tag)
                tagAndWord[tag] = dict()
            vocab.add(word)
            if word in tagAndWord[tag]:
                tagAndWord[tag][word] += 1
            else:
                tagAndWord[tag][word] = 1
            prev = word

    # for line in lines:
    #     prev = "<s>"
    #     sentence = line.rstrip()
    #     sents = sentence.split(" ")
    #     for sent in sents:
    #         word, tag = sent.rsplit("/", 1)
    #         word = processWord(word, tag, prev)
    #         enlargeCorpus(vocab, tagAndWord, word, tag)
    #         prev = word
    trainFile.close()

def unknownVocab(vocab, tagAndWord):
    vocabUNK = set()
    for words in tagAndWord.values():
        cntUNK = 0
        unkSet = set()
        for word, times in words.items():
            if times == 1:
                cntUNK += 1
                unkSet.add(word)
        words["<UNK>"] = cntUNK
        vocabUNK = vocabUNK.union(unkSet)
        for word in unkSet: words.pop(word, None)
        tot_times = sum(words.values())
        cnt_type = len(words.keys())
        smooth_ratio = 0
        for word, times in words.items():
            if words["<UNK>"] == 0:
                words[word] = times/tot_times
            else:
                if word == "<UNK>":
                    words[word] = (times + cnt_type*smooth_ratio) / (tot_times + cnt_type*smooth_ratio)
                else:
                    words[word] = times / (tot_times + cnt_type*smooth_ratio)
    for word in vocabUNK:
        stillStanding = False
        for words in tagAndWord.values():
            if word in words:
                stillStanding = True
                break
        if not stillStanding:
            vocab.discard(word)

def calMatrix(train_file, tags):
    trainFile = open(train_file, "r")
    lines = trainFile.readlines()
    numRow = numCol = len(tags) + 1
    matrix = np.zeros((numRow, numCol), dtype = float)
    tagCol = deepcopy(tags)
    tagCol.append("</s>")
    tagRow = ["<s>"]
    tagRow.extend(deepcopy(tags))
    rowDict = {k : tagRow.index(k) for k in tagRow}
    colDict = {k : tagCol.index(k) for k in tagCol}
    for line in lines:
        prev = "<s>"
        sentence = line.rstrip()
        sents = sentence.split(" ")
        for sent in sents:
            _, tag = sent.rsplit("/", 1)
            matrix[rowDict[prev]][colDict[tag]] += 1
            prev = tag
        matrix[rowDict[prev]][colDict["</s>"]] += 1
    trainFile.close()
    return matrix

def wbsmoothing(matrix, tags):
    cntTags = len(tags)
    for row in range(cntTags + 1):
        T = 0
        Cw0 = 0
        V = cntTags + 1
        for col in range(cntTags + 1):
            if matrix[row][col] != 0:
                T += 1
                Cw0 += matrix[row][col]
        for col in range(cntTags + 1):
            if matrix[row][col] != 0:
                matrix[row][col] /= (Cw0 + T)
            else:
                matrix[row][col] = T / ((V - T) * (Cw0 + T))

def train_model(train_file, model_file):
    vocab = set()
    tagAndWord = dict()
    tags = list()
    loadTrainData(train_file, vocab, tags, tagAndWord)
    unknownVocab(vocab, tagAndWord)
    table = calMatrix(train_file, tags)
    wbsmoothing(table, tags)
    saveModel(model_file, table, vocab, tags, tagAndWord)
    print('Finished...')

if __name__ == "__main__":
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)