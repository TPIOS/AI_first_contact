import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8') 

file = open("cooccurrences.bin","rb")
lines = file.read()

s = lines.decode(errors="ignore")
print(s)
# print(lines)


file.close()