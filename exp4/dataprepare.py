import numpy as np
import json
word2idx={}
idx2word={}
word2idx_file_path="dataset/WikiWord/word2idx.json"
idx2word_file_path="dataset/WikiWord/idx2word.json"

file_list=["dataset/WikiWord/test.txt","dataset/WikiWord/train.txt","dataset/WikiWord/validation.txt"]
next_idx=1
max_seq_len=0
for file_path in file_list:
    with open(file_path,"r") as f:
        lines=f.readlines()
        for line in lines:
            line=line.replace("\t"," ")
            line=line.strip("\n").split(" ")
            if len(line)<2:
                continue
            max_seq_len=max(max_seq_len,len(line))
            for word in line[1:]:
                if word not in word2idx:
                    word2idx[word]=next_idx
                    idx2word[next_idx]=word
                    next_idx+=1

with open(word2idx_file_path,"w",encoding="utf-8") as f:
    json.dump(word2idx,f,ensure_ascii=False)
with open(idx2word_file_path,"w",encoding="utf-8") as f:
    json.dump(idx2word,f,ensure_ascii=False)
print(f"max seq len {max_seq_len}")
print(f"vocabulary size {len(word2idx)}")

