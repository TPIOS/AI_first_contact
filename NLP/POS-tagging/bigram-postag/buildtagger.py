import os
import sys
import datetime
from copy import deepcopy

def write_file(model_file, tagAppearTime, tagBeforeTag, wordAndTag):
    model_file = open(model_file, "w")

    for key, value in tagAppearTime.items(): model_file.write(key + ":" + str(value) + "\n")

    model_file.write("!@#\n")
    for key1 in tagBeforeTag.keys():
        for key2, value in tagBeforeTag[key1].items():
            model_file.write(key1 + " " + key2 + ":" + str(value) + "\n")

    model_file.write("!@#\n")
    for key1 in wordAndTag.keys():
        for key2, value in wordAndTag[key1].items():
            model_file.write(key1 + " " + key2 + ":" + str(value) + "\n")

    model_file.close()

def train_model(train_file, model_file):
    trainFile = open(train_file, "r");

    tagAppearTime = dict()
    tagBeforeTag = dict()
    tagAfterTag = dict()
    wordAndTag = dict()
    
    tagAppearTime["<s>"] = 0
    tagAppearTime["</s>"] = 0

    lines = trainFile.readlines()
    for line in lines:
        prev = "<s>"
        tagAppearTime["<s>"] += 1
        tagAppearTime["</s>"] += 1
        sentence = line.strip()
        words = sentence.split(" ")
        for i in range(len(words)):
            word, tag = words[i].rsplit("/", 1)
            if i == 0 and (tag != "NNP" or tag != "NNPS"): word = word.lower()
            if tag in tagAppearTime:
                tagAppearTime[tag] += 1
            else:
                tagAppearTime[tag] = 1

            if prev in tagBeforeTag:
                if tag in tagBeforeTag[prev]:
                    tagBeforeTag[prev][tag] += 1
                else:
                    tagBeforeTag[prev][tag] = 1
            else:
                tagBeforeTag[prev] = dict()
                tagBeforeTag[prev][tag] = 1
            
            if word in wordAndTag:
                if tag in wordAndTag[word]:
                    wordAndTag[word][tag] += 1
                else:
                    wordAndTag[word][tag] = 1
            else:
                wordAndTag[word] = dict()
                wordAndTag[word][tag] = 1         

            prev = tag

        if prev in tagBeforeTag:
            if "</s>" in tagBeforeTag[prev]:
                tagBeforeTag[prev]["</s>"] += 1
            else:
                tagBeforeTag[prev]["</s>"] = 1
        else:
            tagBeforeTag[prev] = dict()
            tagBeforeTag[prev]["</s>"] = 1

    for line in lines:
        sentence = line.strip()
        words = sentence.split(" ")
        for i in range(len(words)):
            word, tag = words[i].rsplit("/", 1)
            if i == 0 and (tag != "NNP" or tag != "NNPS"): word = word.lower()
            if tag == "NNS":
                if word[:-1] in wordAndTag:
                    if "NN" in wordAndTag[word[:-1]]:
                        wordAndTag[word[:-1]]["NN"] += 1
                    else:
                        wordAndTag[word[:-1]]["NN"] = 1
                if word[:-2] in wordAndTag:
                    if "NN" in wordAndTag[word[:-2]]:
                        wordAndTag[word[:-2]]["NN"] += 1
                    else:
                        wordAndTag[word[:-2]]["NN"] = 1    
                if word[:-3] + "y" in wordAndTag:
                    if "NN" in wordAndTag[word[:-3] + "y"]:
                        wordAndTag[word[:-3] + "y"]["NN"] += 1
                    else:
                        wordAndTag[word[:-3] + "y"]["NN"] = 1

            if tag == "NNPS":
                if word[:-1] in wordAndTag:
                    if "NNP" in wordAndTag[word[:-1]]:
                        wordAndTag[word[:-1]]["NNP"] += 1
                    else:
                        wordAndTag[word[:-1]]["NNP"] = 1
                if word[:-2] in wordAndTag:
                    if "NNP" in wordAndTag[word[:-2]]:
                        wordAndTag[word[:-2]]["NNP"] += 1
                    else:
                        wordAndTag[word[:-2]]["NNP"] = 1    
                if word[:-3] + "y" in wordAndTag:
                    if "NNP" in wordAndTag[word[:-3] + "y"]:
                        wordAndTag[word[:-3] + "y"]["NNP"] += 1
                    else:
                        wordAndTag[word[:-3] + "y"]["NNP"] = 1
            
            if tag == "NN":
                if word+"s" in wordAndTag:
                    if "NNS" in wordAndTag[word+"s"]:
                        wordAndTag[word+"s"]["NNS"] += 1
                    else:
                        wordAndTag[word+"s"]["NNS"] = 1
            
            if tag == "NNP":
                if word+"s" in wordAndTag:
                    if "NNPS" in wordAndTag[word+"s"]:
                        wordAndTag[word+"s"]["NNPS"] += 1
                    else:
                        wordAndTag[word+"s"]["NNPS"] = 1
        
    write_file(model_file, tagAppearTime, tagBeforeTag, wordAndTag)
    print('Finished...')
    trainFile.close()

if __name__ == "__main__":
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)