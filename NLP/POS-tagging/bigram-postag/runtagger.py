import os
import math
import sys
import datetime
from copy import deepcopy
import numpy as np
import re, string

def loadModel(model_file):
    modelFile = open(model_file, "r")
    tagAppearTime = dict()
    tagBeforeTag = dict()
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
            if firstKey in wordAndTag:
                wordAndTag[firstKey][secondKey] = times
            else:
                wordAndTag[firstKey] = dict()
                wordAndTag[firstKey][secondKey] = times
    return tagAppearTime, tagBeforeTag, wordAndTag

def is_number(num):
    if num == "'" or num == "'s": return False
    pattern = re.compile(r'^[-+]?[\']?([0-9]+[.,-/]?)*[s]?$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False

def processWord(x, localWordAndTag):
    if is_number(x): #for all number, it should be number
        localWordAndTag[x] = dict()
        localWordAndTag[x]["CD"] = 100
        return x

    #end-with-ing
    if x.endswith("ing"):
        if x in localWordAndTag:
            if not "VBG" in localWordAndTag[x]: localWordAndTag[x]["VBG"] = 10
            if not "NN" in localWordAndTag[x]: localWordAndTag[x]["NN"] = 10
        else:
            localWordAndTag[x] = dict()
            localWordAndTag[x]["VBG"] = 10
            localWordAndTag[x]["NN"] = 10
            localWordAndTag[x]["JJ"] = 10
        return x

    #end-with-ment/ness
    if x.endswith("ment") or x.endswith("ness"):
        if x in localWordAndTag:
            if not "NN" in localWordAndTag[x]: localWordAndTag["NN"] = 10
        else:
            localWordAndTag[x] = dict()
            localWordAndTag[x]["NN"] = 10
        return x
        
    #capitalized word
    if x == x.capitalize():
        if x.endswith("s"):
            if not x in localWordAndTag:
                localWordAndTag[x] = dict()
                localWordAndTag[x]["NNPS"] = 20
            else:
                if not "NNPS" in localWordAndTag[x]: localWordAndTag[x]["NNPS"] = 10
                if not "JJ" in localWordAndTag[x]: localWordAndTag["JJ"] = 10
        else:
            if not x in localWordAndTag:
                localWordAndTag[x] = dict()
                localWordAndTag[x]["NNP"] = 20
            else:
                if not "NNP" in localWordAndTag[x]: localWordAndTag[x]["NNP"] = 10
                if not "JJ" in localWordAndTag[x]: localWordAndTag["JJ"] = 10
        return x

    #abbr.
    if x == x.upper():
        if not x in localWordAndTag:
            localWordAndTag[x] = dict()
            localWordAndTag[x]["NNP"] = 20

    #dash
    if "-" in x:
        if not x in localWordAndTag:
            localWordAndTag[x] = dict()
            localWordAndTag[x]["NNP"] = 10
            localWordAndTag[x]["JJ"] = 10
            localWordAndTag[x]["NN"] = 10
        return x

    if x in localWordAndTag: return x # not UNK, just return

    if x.capitalize() in localWordAndTag or x.lower() in localWordAndTag or x.upper() in localWordAndTag:
        if x.capitalize() in localWordAndTag:
            localWordAndTag[x] = deepcopy(localWordAndTag[x.capitalize()]) #enlarge corpus
        if x.lower() in localWordAndTag:
            localWordAndTag[x] = deepcopy(localWordAndTag[x.lower()]) #enlarge corpus
        if x.upper() in localWordAndTag:
            localWordAndTag[x] = deepcopy(localWordAndTag[x.upper()]) #enlarge corpus
        return x

    #end-with-s
    if x.endswith("s") and x[:-1] in localWordAndTag:
        if not x in localWordAndTag: localWordAndTag[x] = dict()
        if not "NNS" in localWordAndTag[x]: localWordAndTag[x]["NNS"] = 0
        if not "NNPS" in localWordAndTag[x]: localWordAndTag[x]["NNPS"] = 0
        if not "VBZ" in localWordAndTag[x]: localWordAndTag[x]["VBZ"] = 0
        for key in localWordAndTag[x[:-1]]:
            if key == "NN": localWordAndTag[x]["NNS"] += localWordAndTag[x[:-1]]["NN"]
            if key == "NNP": localWordAndTag[x]["NNPS"] += localWordAndTag[x[:-1]]["NNP"]
            if key == "VBP": localWordAndTag[x]["VBZ"] += localWordAndTag[x[:-1]]["VBP"]
            if key == "VB": localWordAndTag[x]["VBZ"] += localWordAndTag[x[:-1]]["VB"]
        return x
    elif x.endswith("s"):
        localWordAndTag[x] = dict()
        localWordAndTag[x]["NNS"] = 20
        localWordAndTag[x]["NNPS"] = 20
    
    # #single-plural
    if x+"s" in localWordAndTag:
        localWordAndTag[x] = dict()
        for key in localWordAndTag[x+"s"]:
            if key == "NNS": localWordAndTag[x]["NN"] = localWordAndTag[x+"s"]["NNS"]
            if key == "NNPS": localWordAndTag[x]["NNP"] = localWordAndTag[x+"s"]["NNPS"]
            if key == "VBZ":
                localWordAndTag[x]["VB"] = localWordAndTag[x+"s"]["VBZ"]
                localWordAndTag[x]["VBP"] = localWordAndTag[x+"s"]["VBZ"]
        return x
    
    #pass tone
    if x.endswith("d") and x[:-1] in localWordAndTag:
        if not x in localWordAndTag: localWordAndTag[x] = dict()
        if not "VBD" in localWordAndTag[x]: localWordAndTag[x]["VBD"] = 0
        if not "VBN" in localWordAndTag[x]: localWordAndTag[x]["VBN"] = 0
        for key in localWordAndTag[x[:-1]]:
            if key == "VBP":
                localWordAndTag[x]["VBD"] += localWordAndTag[x[:-1]]["VBP"]
                localWordAndTag[x]["VBN"] += localWordAndTag[x[:-1]]["VBP"]
            if key == "VB":
                localWordAndTag[x]["VBD"] += localWordAndTag[x[:-1]]["VB"]
                localWordAndTag[x]["VBN"] += localWordAndTag[x[:-1]]["VB"]
        return x
    
    if x.endswith("ed") and x[:-2] in localWordAndTag:
        if not x in localWordAndTag: localWordAndTag[x] = dict()
        if not "VBD" in localWordAndTag[x]: localWordAndTag[x]["VBD"] = 0
        if not "VBN" in localWordAndTag[x]: localWordAndTag[x]["VBN"] = 0
        for key in localWordAndTag[x[:-2]]:
            if key == "VBP":
                localWordAndTag[x]["VBD"] += localWordAndTag[x[:-2]]["VBP"]
                localWordAndTag[x]["VBN"] += localWordAndTag[x[:-2]]["VBP"]
            if key == "VB":
                localWordAndTag[x]["VBD"] += localWordAndTag[x[:-2]]["VB"]
                localWordAndTag[x]["VBN"] += localWordAndTag[x[:-2]]["VB"]
        return x

def viterbi(localTagAppearTime, localTagAndTag, localWordAndTag, numOfWords, numOfTags, words, Tags, startPos):
    table = np.zeros((numOfWords, numOfTags))
    backpoint = [[-1 for i in range(numOfTags)] for j in range(numOfWords)]
    tempMax = 0
    argMax = -1
    if startPos == "<s>": endPos = "</s>"

    words[0] = words[0].lower()
    words[0] = processWord(words[0], localWordAndTag) #assume we just add the times that UNK as a Tag but not add Tag appear time.
    for i in range(numOfTags):
        if words[0] in localWordAndTag.keys():
            if Tags[i] in localTagAndTag[startPos] and Tags[i] in localWordAndTag[words[0]]:
                table[0][i] = (localTagAndTag[startPos][Tags[i]] / localTagAppearTime[startPos]) * (localWordAndTag[words[0]][Tags[i]] / localTagAppearTime[Tags[i]])
        else:
            if Tags[i] in localTagAndTag[startPos]:
                table[0][i] = (localTagAndTag[startPos][Tags[i]] / localTagAppearTime[startPos])

    # print(table[0])

    preWord = words[0]
    for i in range(1, numOfWords):
        if preWord == "``" or preWord == "''" or preWord == "(": words[i] = words[i].lower()
        words[i] = processWord(words[i], localWordAndTag)
        # if words[i] == "1990": print(localWordAndTag["1990"])
        for j in range(numOfTags):
            for k in range(numOfTags):
                if words[i] in localWordAndTag.keys():
                    if Tags[j] in localTagAndTag[Tags[k]] and Tags[j] in localWordAndTag[words[i]]: 
                        cal = (localTagAndTag[Tags[k]][Tags[j]] / localTagAppearTime[Tags[k]]) * (localWordAndTag[words[i]][Tags[j]] / localTagAppearTime[Tags[j]]) * table[i-1][k]
                        if cal > table[i][j]:
                            table[i][j] = cal
                            backpoint[i][j] = k
                else:
                    if Tags[j] in localTagAndTag[Tags[k]]:
                        cal = (localTagAndTag[Tags[k]][Tags[j]] / localTagAppearTime[Tags[k]]) * table[i-1][k]
                        if cal > table[i][j]:
                            table[i][j] = cal
                            backpoint[i][j] = k
        allZero = True
        for j in range(numOfTags):
            if table[i][j] != 0: 
                allZero = False
                break
        if allZero:
            for j in range(numOfTags):
                for k in range(numOfTags):
                    if Tags[j] in localTagAndTag[Tags[k]]:
                        cal = (localTagAndTag[Tags[k]][Tags[j]] / localTagAppearTime[Tags[k]]) * table[i-1][k]
                        if cal > table[i][j]:
                            table[i][j] = cal
                            backpoint[i][j] = k

        preWord = words[i]
        # print(table[i])
    
    for i in range(numOfTags):
        if endPos in localTagAndTag[Tags[i]]:
            cal = (localTagAndTag[Tags[i]][endPos] / localTagAppearTime[Tags[i]]) * table[numOfWords-1][i]
            if cal > tempMax:
                tempMax = cal
                argMax = i

    tag = list()
    for i in range(numOfWords-1, -1, -1):
        tag.append(Tags[argMax])
        argMax = backpoint[i][argMax]

    if startPos == "<s>": tag = tag[::-1]
    return table, tag

def algorithm(words, tagAppearTime, tagBeforeTag, wordAndTag):
    localTagAppearTime = deepcopy(tagAppearTime)
    localTagBeforeTag = deepcopy(tagBeforeTag)
    localWordAndTag = deepcopy(wordAndTag)
    localTagBeforeTag["</s>"] = dict()
    numOfWords = len(words)
    Tags = list(localTagAppearTime.keys())
    numOfTags = len(localTagAppearTime.keys())
    forwardTable, forwardTag = viterbi(localTagAppearTime, localTagBeforeTag, localWordAndTag, numOfWords, numOfTags, words, Tags, "<s>")
    return forwardTag

def tag_sentence(test_file, model_file, out_file):
    tagAppearTime, tagBeforeTag, wordAndTag = loadModel(model_file)
    testFile = open(test_file, "r")
    outFile = open(out_file, "w")
    lines = testFile.readlines()
    for line in lines:
        sentence = line.rstrip()
        words = sentence.split(" ")
        tag = algorithm(words, tagAppearTime, tagBeforeTag, wordAndTag)
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