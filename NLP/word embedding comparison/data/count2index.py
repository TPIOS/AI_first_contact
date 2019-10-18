file = open("enwiki2018_c50k.count","r")
idx = open("enwiki2018_c50k.index","w")

lines = file.readlines()
cnt = 1
for line in lines:
    word, times = line.split(" ")
    newline = str(cnt)+" "+word+'\n'
    idx.write(newline)
    cnt += 1

print(cnt)

idx.close()
file.close()