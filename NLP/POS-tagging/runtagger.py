import os
import math
import sys
import datetime
from copy import deepcopy
import numpy as np

def loadModel(model_file):
    modelFile = open(model_file, "r")
    tagAppearTime = dict()
    tagBeforeTag = dict()
    tagAfterTag = dict()
    wordAndTag = dict()
    lines = modelFile.readlines()
    modelFile.close()
    whichDict = 0
    for line in lines:
        data = line.rstrip()
        if data == "!@#":
            whichDict += 1
            continue
        manyKeys, times = data.rsplit(":", 1)
        times = int(times)
        if whichDict == 0:
            tagAppearTime[manyKeys] = times
            continue
        firstKey, secondKey = manyKeys.split(" ")
        if whichDict == 1:
            if firstKey in tagBeforeTag:
                tagBeforeTag[firstKey][secondKey] = times
            else:
                tagBeforeTag[firstKey] = dict()
                tagBeforeTag[firstKey][secondKey] = times
        if whichDict == 2:
            if firstKey in tagAfterTag:
                tagAfterTag[firstKey][secondKey] = times
            else:
                tagAfterTag[firstKey] = dict()
                tagAfterTag[firstKey][secondKey] = times
        if whichDict == 3:
            if firstKey in wordAndTag:
                wordAndTag[firstKey][secondKey] = times
            else:
                wordAndTag[firstKey] = dict()
                wordAndTag[firstKey][secondKey] = times
    return tagAppearTime, tagBeforeTag, tagAfterTag, wordAndTag

def viterbi(words, tagAppearTime, tagBeforeTag, tagAfterTag, wordAndTag, smoothing = False):
    localTagAppearTime = deepcopy(tagAppearTime)
    localTagBeforeTag = deepcopy(tagBeforeTag)
    localTagAfterTag = deepcopy(tagAfterTag)
    localWordAndTag = deepcopy(wordAndTag)
    localTagBeforeTag["</s>"] = dict()
    localTagAfterTag["<s>"] = dict()
    tag = list()
    numOfWords = len(words)
    Tags = list(localTagAppearTime.keys())
    numOfTags = len(localTagAppearTime.keys())
    table = np.zeros((numOfWords, numOfTags))
    backpoint = [[-1 for i in range(numOfTags)] for j in range(numOfWords)]
    tempMax = 0
    argMax = -1
    if not smoothing:
        for i in range(numOfTags):
            if words[0] in localWordAndTag.keys():
                if Tags[i] in localTagBeforeTag["<s>"] and Tags[i] in localWordAndTag[words[0]]:
                    table[0][i] = (localTagBeforeTag["<s>"][Tags[i]] / localTagAppearTime["<s>"]) * (localWordAndTag[words[0]][Tags[i]] / localTagAppearTime[Tags[i]])
            else:
                if Tags[i] in localTagBeforeTag["<s>"]:
                    table[0][i] = (localTagBeforeTag["<s>"][Tags[i]] / localTagAppearTime["<s>"])
        for i in range(1, numOfWords):
            for j in range(numOfTags):
                for k in range(numOfTags):
                    if words[i] in localWordAndTag.keys():
                        if Tags[j] in localTagBeforeTag[Tags[k]] and Tags[j] in localWordAndTag[words[i]]: 
                            cal = (localTagBeforeTag[Tags[k]][Tags[j]] / localTagAppearTime[Tags[k]]) * (localWordAndTag[words[i]][Tags[j]] / localTagAppearTime[Tags[j]]) * table[i-1][k]
                            if cal > table[i][j]:
                                table[i][j] = cal
                                backpoint[i][j] = k
                    else:
                        if Tags[j] in localTagBeforeTag[Tags[k]]:
                            cal = (localTagBeforeTag[Tags[k]][Tags[j]] / localTagAppearTime[Tags[k]]) * table[i-1][k]
                            if cal > table[i][j]:
                                table[i][j] = cal
                                backpoint[i][j] = k
        for i in range(numOfTags):
            if "</s>" in localTagBeforeTag[Tags[i]]:
                cal = (localTagBeforeTag[Tags[i]]["</s>"] / localTagAppearTime[Tags[i]]) * table[numOfWords-1][i]
                if cal > tempMax:
                    tempMax = cal
                    argMax = i
        
    for i in range(numOfWords-1, -1, -1):
        tag.append(Tags[argMax])
        argMax = backpoint[i][argMax]
    tag = tag[::-1]
    assert(len(tag) == len(words))
    return tag

def tag_sentence(test_file, model_file, out_file):
    tagAppearTime, tagBeforeTag, tagAfterTag, wordAndTag = loadModel(model_file)
    testFile = open(test_file, "r")
    outFile = open(out_file, "w")
    lines = testFile.readlines()
    for line in lines:
        sentence = line.rstrip()
        words = sentence.lower().split(" ")
        tag = viterbi(words, tagAppearTime, tagBeforeTag, tagAfterTag, wordAndTag)
        words = sentence.split(" ")
        ans = list()
        for i in range(len(tag)): ans.append(words[i]+"/"+tag[i])
        ans = " ".join(ans)
        outFile.write(ans+"\n")

    print('Finished...')
    testFile.close()
    outFile.close()  

if __name__ == "__main__":
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)