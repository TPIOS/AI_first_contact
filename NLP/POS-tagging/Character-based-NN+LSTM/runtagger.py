import os
import math
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CharacterBasedCNNBiLSTM(nn.Module):
    def __init__(self, vocabSize, tagSize, charSize, characterEmbeddingDim, wordEmbeddingDim, hiddenDim, NumCNNLayer, windowSize, NumHiddenLayer, wordPaddingIdx):
        super(CharacterBasedCNNBiLSTM, self).__init__()
        self.hidden_dim = hiddenDim
        # self.num_CNN_layer = NumCNNLayer
        # self.num_hidder_layer = NumHiddenLayer
        # self.char_embeddings = nn.Embedding(charSize, characterEmbeddingDim).to(device)
        # self.CNN = nn.Conv1d(characterEmbeddingDim, NumCNNLayer, windowSize, padding = 1).to(device)
        self.word_embeddings = nn.Embedding(vocabSize, wordEmbeddingDim, padding_idx = wordPaddingIdx).to(device)
        self.LSTM = nn.LSTM(wordEmbeddingDim + NumCNNLayer, hiddenDim, num_layers = 4, dropout=0.5, batch_first = True, bidirectional=True).to(device)
        self.hidden_to_tag = nn.Linear(hiddenDim * 2, tagSize).to(device)

    def init_idx(self, char2idx, word2idx, tag2idx, idx2tag, unk):
        self.char2idx = char2idx
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
        self.unk = unk

    def init_hidden(self, batchSize):
        self.hidden = (torch.randn(self.num_hidder_layer * 2, batchSize, self.hidden_dim).to(device),
                       torch.randn(self.num_hidder_layer * 2, batchSize, self.hidden_dim).to(device))

    def prepare_sequence(self, words):
        char_tensors = []
        word_tensors = []
        word_lengths = []
        for word in words:
            if word in self.word2idx:
                word_tensors.append(self.word2idx[word])
            else:
                word_tensors.append(self.word2idx[self.unk])
            word_lengths.append(len(word))

        max_word_length = max(word_lengths)
        for word in words:
            char_idx = []
            word_len = len(word)
            for char in word:
                if char in self.char2idx:
                    char_idx.append(self.char2idx[char])
                else:
                    char_idx.append(self.char2idx[self.unk])
            char_idx += [self.char2idx["<PAD>"]] * (max_word_length - word_len)
            char_idx = torch.tensor(char_idx, dtype=torch.long).to(device)
            char_tensors.append(char_idx)

        char_tensors = torch.stack(char_tensors) # dim: word_len * char_len
        word_tensors = torch.tensor(word_tensors, dtype=torch.long).to(device) # dim: word_len
        return char_tensors, word_tensors

    def forward(self, charTensors, wordTensors):
        char_embeddings = self.char_embeddings(charTensors)
        batch_size = 1
        num_words = char_embeddings.shape[0]
        num_chars = char_embeddings.shape[1]
        char_dimension = char_embeddings.shape[2]
        char_embeddings = char_embeddings.view(batch_size * num_words, char_dimension, num_chars)
        char_CNN_out = self.CNN(char_embeddings)
        char_CNN_out, _ = torch.max(char_CNN_out, 2)
        char_CNN_out = char_CNN_out.view(batch_size, num_words, self.num_CNN_layer)

        word_embeddings = self.word_embeddings(wordTensors)
        word_embeddings = torch.unsqueeze(word_embeddings, 0)
        word_embeddings = torch.cat((word_embeddings, char_CNN_out), 2)
        LSTM_out, self.hidden = self.LSTM(word_embeddings, self.hidden)
        LSTM_out = LSTM_out.contiguous()
        tag_space = self.hidden_to_tag(LSTM_out.view(-1, LSTM_out.shape[2]))
        tag_scores = F.log_softmax(tag_space, dim=1).to(device)
        return tag_scores

def tag_sentence(test_file, model_file, out_file):
    startTime = datetime.datetime.now()
    model = torch.load(model_file)
    if torch.cuda.is_available(): model.cuda()
    testFile = open(test_file, "r")
    lines = testFile.readlines()
    testSentences = []
    for line in lines: testSentences.append(line.rstrip())
    
    outputSentences = []
    for sentence in testSentences:
        with torch.no_grad():
            words = sentence.split(" ")
            charTensor, wordTensor = model.prepare_sequence(words)
            tagScores = model(charTensor, wordTensor)
            resSentence = []
            for i in range(len(words)):
                word = words[i]
                _, tagIdx = tagScores[i].max(0)
                tag = model.idx2tag[tagIdx.item()]
                resSentence.append(word + "/" + tag)
            outputSentences.append(" ".join(resSentence))
    
    outFile = open(out_file, "w")
    outFile.write("\n".join(outputSentences))
    testFile.close()
    outFile.close()

    endTime = datetime.datetime.now()
    print('Cost:', endTime - startTime)
    print('Finished...')

if __name__ == "__main__":
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)