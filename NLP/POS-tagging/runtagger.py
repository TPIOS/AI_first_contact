import os
import math
import sys
import datetime

def loadModel(model_file):
    modelFile = open(model_file, "r")
    tagAppearTime = dict()
    tagBeforeTag = dict()
    tagAfterTag = dict()
    wordAndTag = dict()
    lines = modelFile.readlines()
    modelFile.close()
    whichDict = 0
    for line in lines():
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
            tagBeforeTag[firstKey] = dict()
            tagBeforeTag[firstKey][secondKey] = times
        if whichDict == 2:
            tagAfterTag[firstKey] = dict()
            tagAfterTag[firstKey][secondKey] = times
        if whichDict == 3:
            wordAndTag[firstKey] = dict()
            wordAndTag[firstKey][secondKey] = times

    return tagAppearTime, tagBeforeTag, tagAfterTag, wordAndTag

def viterbi(words, tagAppearTime, tagBeforeTag, tagAfterTag, wordAndTag):
    localTagAppearTime = tagAppearTime.copy()
    localTagBeforeTag = tagBeforeTag.copy()
    localTagAfterTag = tagAfterTag.copy()
    localWordAndTag = wordAndTag.copy()
    tag = list()
    for word in words:
        if word in localWordAndTag:
            for key in localTagAppearTime.keys():
                if key in localWordAndTag[word]:
                    localWordAndTag[word][key] += 1
                    localTagAppearTime[key] += 1
                else:
                    localWordAndTag[word][key] = 1
                    localTagAppearTime[key] += 1
        else:
            localWordAndTag[word] = dict()
            for key in localTagAppearTime.keys():
                localTagAppearTime[key] += 1
                localWordAndTag[word][key] = 1

    
                
            
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