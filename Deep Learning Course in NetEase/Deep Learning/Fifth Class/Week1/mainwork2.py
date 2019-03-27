import numpy as np
from utils import *
import random
from di import *

data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
# print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
# print(ix_to_char)

# np.random.seed(3)
# dWax = np.random.randn(5,3)*10
# dWaa = np.random.randn(5,5)*10
# dWya = np.random.randn(2,5)*10
# db = np.random.randn(5,1)*10
# dby = np.random.randn(2,1)*10
# gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
# gradients = clip(gradients, 10)
# print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
# print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
# print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
# print("gradients[\"db\"][4] =", gradients["db"][4])
# print("gradients[\"dby\"][1] =", gradients["dby"][1])

# np.random.seed(2)
# _, n_a = 20, 100
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


# indices = sample(parameters, char_to_ix, 0)
# print("Sampling:")
# print("list of sampled indices:", indices)
# print("list of sampled characters:", [ix_to_char[i] for i in indices])
# np.random.seed(1)
# vocab_size, n_a = 27, 100
# a_prev = np.random.randn(n_a, 1)
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
# X = [12,3,5,11,22,3]
# Y = [4,14,11,22,25, 26]

# loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
# print("Loss =", loss)
# print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
# print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
# print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
# print("gradients[\"db\"][4] =", gradients["db"][4])
# print("gradients[\"dby\"][1] =", gradients["dby"][1])
# print("a_last[4] =", a_last[4])

parameters = model(data, ix_to_char, char_to_ix)