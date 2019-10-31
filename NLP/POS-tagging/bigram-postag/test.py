import string
testFile = open("sents.test", "r")
lines = testFile.readlines()
for line in lines:
    sentence = line.rstrip()
    if sentence[-1] not in string.punctuation:
        print(sentence)
testFile.close()