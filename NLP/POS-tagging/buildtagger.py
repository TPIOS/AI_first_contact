import os
import sys
import datetime

def write_file(model_file, tagAppearTime, tagBeforeTag, tagAfterTag, wordAndTag):
    model_file = open(model_file, "w")

    for key, value in tagAppearTime.items(): model_file.write(key + ":" + str(value) + "\n")

    model_file.write("!@#\n")
    for key1 in tagBeforeTag.keys():
        for key2, value in tagBeforeTag[key1].items():
            model_file.write(key1 + " " + key2 + ":" + str(value) + "\n")
    
    model_file.write("!@#\n")
    for key1 in tagAfterTag.keys():
        for key2, value in tagAfterTag[key1].items():
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
        for wordWithTag in words:
            word, tag = wordWithTag.rsplit("/", 1)
            word = word.lower()
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
        
        late = "</s>"
        for wordWithTag in words[::-1]:
            word, tag = wordWithTag.rsplit("/", 1)
            word = word.lower()
            if late in tagAfterTag:
                if tag in tagAfterTag[late]:
                    tagAfterTag[late][tag] += 1
                else:
                    tagAfterTag[late][tag] = 1
            else:
                tagAfterTag[late] = dict()
                tagAfterTag[late][tag] = 1
            late = tag

        if late in tagAfterTag:
            if "<s>" in tagAfterTag[late]:
                tagAfterTag[late]["<s>"] += 1
            else:
                tagAfterTag[late]["<s>"] = 1
        else:
            tagAfterTag[late] = dict()
            tagAfterTag[late]["<s>"] = 1
        
    write_file(model_file, tagAppearTime, tagBeforeTag, tagAfterTag, wordAndTag)
    print('Finished...')
    trainFile.close()

if __name__ == "__main__":
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)