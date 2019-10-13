import os
import sys
import datetime

def write_file(model_file, tagAppearTime, tagAndTag, wordAndTag):
    model_file = open(model_file, "w")
    for key, value in tagAppearTime.items(): model_file.write(key + ":" + str(value) + "\n")
    for key1 in tagAndTag.keys():
        for key2, value in tagAndTag[key1].items():
            model_file.write(key1 + " " + key2 + ":" + str(value) + "\n")
    for key1 in wordAndTag.keys():
        for key2, value in wordAndTag[key1].items():
            model_file.write(key1 + " " + key2 + ":" + str(value) + "\n")
    model_file.close()

def train_model(train_file, model_file):
    trainFile = open(train_file, "r");

    tagAppearTime = dict()
    tagAndTag = dict()
    wordAndTag = dict()

    lines = trainFile.readlines()
    for line in lines:
        prev = "<s>"
        sentence = line.strip()
        words = sentence.split(" ")
        for wordWithTag in words:
            word, tag = wordWithTag.rsplit("/", 1)
            word = word.lower()
            if tag in tagAppearTime:
                tagAppearTime[tag] += 1
            else:
                tagAppearTime[tag] = 1

            if prev in tagAndTag:
                if tag in tagAndTag[prev]:
                    tagAndTag[prev][tag] += 1
                else:
                    tagAndTag[prev][tag] = 1
            else:
                tagAndTag[prev] = dict()
                tagAndTag[prev][tag] = 1
            
            if word in wordAndTag:
                if tag in wordAndTag[word]:
                    wordAndTag[word][tag] += 1
                else:
                    wordAndTag[word][tag] = 1
            else:
                wordAndTag[word] = dict()
                wordAndTag[word][tag] = 1         

            prev = tag
        
        if prev in tagAndTag:
            if "</s>" in tagAndTag[prev]:
                tagAndTag[prev]["</s>"] += 1
            else:
                tagAndTag[prev]["</s>"] = 1
        else:
            tagAndTag[prev] = dict()
            tagAndTag[prev]["</s>"] = 1
        
    write_file(model_file, tagAppearTime, tagAndTag, wordAndTag)
    print('Finished...')
    trainFile.close()

if __name__ == "__main__":
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)