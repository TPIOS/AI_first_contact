import os
import math
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainData:
    def __init__(self, trainFile):
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        trainFileData = open(trainFile, "r")
        lines = trainFileData.readlines()
        self.trainData = []
        self.trainWords = []
        self.trainTags = []
        for line in lines:
            sentence = line.rstrip()
            sents = sentence.split(" ")
            onelineWords = []
            onelineTags = []
            for sent in sents:
                word, tag = sent.rsplit("/", 1)
                onelineWords.append(word)
                onelineTags.append(tag)
                self.trainWords.append(word)
                self.trainTags.append(tag)
            self.trainData.append((onelineWords, onelineTags))
        uniqueWords = set(self.trainWords)
        uniqueWords.add(self.unk)
        uniqueWords.add(self.pad)
        uniqueTags = set(self.trainTags)
        uniqueTags.add(self.pad)

        trainFileData.seek(0)
        text = trainFileData.read()
        text.replace("\n", "")
        text.replace(" ", "")
        character = list(set(text))
        character.append(self.unk)
        character.insert(0, self.pad)
        self.char2idx = {char : i for i, char in enumerate(character)}
        self.word2idx = {word : i for i, word in enumerate(uniqueWords)}
        self.tag2idx = {tag : i for i, tag in enumerate(uniqueTags)}
        self.idx2tag = {i : tag for i, tag in enumerate(uniqueTags)}
        self.wordPaddingIdx = self.word2idx["<PAD>"]
        self.tagPaddingIdx = self.tag2idx["<PAD>"]
        self.charSize = len(self.char2idx)
        self.vocabSize = len(self.word2idx)
        self.tagSize = len(self.tag2idx)

class TrainDataSet(Dataset):
    def __init__(self, trainData):
        self.sentences = []
        self.labels = []
        self.corpus = trainData
        for onelineWords, onelineTags in trainData:
            self.sentences.append(" ".join(onelineWords))
            self.labels.append(" ".join(onelineTags))
    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]
    def __len__(self):
        return len(self.corpus)

class CharacterBasedCNNBiLSTM(nn.Module):
    def __init__(self, vocabSize, tagSize, charSize, characterEmbeddingDim, wordEmbeddingDim, hiddenDim, NumCNNLayer, windowSize, NumHiddenLayer, wordPaddingIdx):
        super(CharacterBasedCNNBiLSTM, self).__init__()
        self.hidden_dim = hiddenDim
        self.num_CNN_layer = NumCNNLayer
        self.num_hidden_layer = NumHiddenLayer
        self.char_embeddings = nn.Embedding(charSize, characterEmbeddingDim).to(device)
        self.CNN = nn.Conv1d(characterEmbeddingDim, NumCNNLayer, windowSize, padding = 1).to(device)
        self.word_embeddings = nn.Embedding(vocabSize, wordEmbeddingDim, padding_idx = wordPaddingIdx).to(device)
        self.LSTM = nn.LSTM(wordEmbeddingDim + NumCNNLayer, hiddenDim, num_layers = NumHiddenLayer, dropout=0.5, batch_first = True, bidirectional=True).to(device)
        self.hidden_to_tag = nn.Linear(hiddenDim * 2, tagSize).to(device)

    def init_idx(self, char2idx, word2idx, tag2idx, idx2tag, unk):
        self.char2idx = char2idx
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
        self.unk = unk

    def init_hidden(self, batchSize):
        self.hidden = (torch.randn(self.num_hidden_layer * 2, batchSize, self.hidden_dim).to(device),
                       torch.randn(self.num_hidden_layer * 2, batchSize, self.hidden_dim).to(device))

    def forward(self, wordTensors, charTensors):
        char_embeddings = self.char_embeddings(charTensors)
        batch_size = char_embeddings.shape[0]
        num_words = char_embeddings.shape[1]
        num_chars = char_embeddings.shape[2]
        char_dimension = char_embeddings.shape[3]
        char_embeddings = char_embeddings.view(batch_size * num_words, char_dimension, num_chars)
        char_CNN_out = self.CNN(char_embeddings)
        char_CNN_out, _ = torch.max(char_CNN_out, 2)
        char_CNN_out = char_CNN_out.view(batch_size, num_words, self.num_CNN_layer)

        word_embeddings = self.word_embeddings(wordTensors)
        # word_embeddings = torch.unsqueeze(word_embeddings, 0)
        word_embeddings = torch.cat((word_embeddings, char_CNN_out), 2).to(device)
        LSTM_out, self.hidden = self.LSTM(word_embeddings, self.hidden)
        LSTM_out = LSTM_out.contiguous()
        tag_space = self.hidden_to_tag(LSTM_out.view(-1, LSTM_out.shape[2]))
        tag_scores = F.log_softmax(tag_space, dim=1).to(device)
        return tag_scores

    def prepare_batch(self, trainData):
        sentence_lengths = []
        word_lengths = []
        sentences, tags = trainData
        max_word_length = -1
        for sentence in sentences: 
            sentence_lengths.append(len(sentence.split(" ")))
            for word in sentence.split(): max_word_length = max(max_word_length, len(word))
        max_sentence_length = max(sentence_lengths)
        
        pad = "<PAD>"
        char_tensors = []
        word_tensors = []
        tag_tensors = []
        for i in range(len(sentence_lengths)):
            padded_sentence = sentences[i].split(" ")
            padded_tag = tags[i].split(" ")
            padded_sentence += [pad] * (max_sentence_length - sentence_lengths[i])
            padded_tag += [pad] * (max_sentence_length - sentence_lengths[i])

            char_idx = []
            sentence_char_idx = []
            for word in padded_sentence:
                if word == pad:
                    char_idx = np.zeros(max_word_length)
                else:
                    char_idx = ([self.char2idx[char] for char in word])
                    char_idx += [0] * (max_word_length - len(word))
                sentence_char_idx.append(torch.tensor(char_idx, dtype=torch.long).to(device))
            
            sentence_char_idx = torch.stack(sentence_char_idx).to(device)
            char_tensors.append(sentence_char_idx)
            word_idx = ([self.word2idx[word] for word in padded_sentence])
            word_tensors.append(torch.tensor(word_idx, dtype=torch.long).to(device))
            tag_idx = ([self.tag2idx[tag] for tag in padded_tag])
            tag_tensors.append(torch.tensor(tag_idx, dtype=torch.long).to(device))
        
        char_tensors = torch.stack(char_tensors).to(device)
        word_tensors = torch.stack(word_tensors).to(device)
        tag_tensors = torch.stack(tag_tensors).to(device)
        return char_tensors, word_tensors, tag_tensors

def train_model(train_file, model_file):
    startTime = datetime.datetime.now()

    characterEmbeddingDim = 10
    windowSize = 3
    NumCNNLayer = 32
    
    wordEmbeddingDim = 300
    hiddenDim = 200
    NumHiddenLayer = 2

    maxEpoch = 5
    batchSize = 16
    # lr = 0.0001
    # momentum = 0.9

    trainData = TrainData(train_file)
    model = CharacterBasedCNNBiLSTM(trainData.vocabSize, trainData.tagSize, trainData.charSize, characterEmbeddingDim, wordEmbeddingDim, hiddenDim, NumCNNLayer, windowSize, NumHiddenLayer, trainData.wordPaddingIdx)
    if torch.cuda.is_available(): model.cuda()
    model.init_idx(trainData.char2idx, trainData.word2idx, trainData.tag2idx, trainData.idx2tag, trainData.unk)

    loss_function = nn.NLLLoss(ignore_index=trainData.tagPaddingIdx).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters())

    trainDataSet = TrainDataSet(trainData.trainData)
    trainDataLoader = DataLoader(trainDataSet, batch_size=batchSize, shuffle=True, num_workers=16)

    endTime = datetime.datetime.now()
    print("Preprocess Cost:", endTime - startTime)

    for epoch in range(maxEpoch):
        print("Epoch:", epoch)
        for iter, data in enumerate(trainDataLoader):
            batch_size = len(data[0])
            model.zero_grad()
            model.init_hidden(batch_size)
            charTensors, wordTensors, tagTensor = model.prepare_batch(data)
            tagTensor = tagTensor.view(-1)

            tagScores = model(wordTensors, charTensors)
            loss = loss_function(tagScores, tagTensor)
            loss.backward()
            optimizer.step()
        endTime = datetime.datetime.now()
        print(endTime - startTime)

    model.init_hidden(1)
    torch.save(model, model_file)

    endTime = datetime.datetime.now()
    print('Cost:', endTime - startTime)
    print('Finished...')
		
if __name__ == "__main__":
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)