import torch
from torch.utils.data import Dataset
import numpy as np
import random
class Text_Dataset(Dataset):
    def __init__(self,file_path,word2idx,max_seq_len=700,augment=False):
        super().__init__()
        self.encoded_lines=[]
        self.targets=[]
        self.augment=augment
        self.max_seq_len=max_seq_len
        with open(file_path,"r") as f:
            lines=f.readlines()
            for line in lines:
                line=line.replace("\t"," ")
                line=line.strip("\n").split(" ")
                if len(line)<2:
                    continue
                self.targets.append(eval(line[0]))
                encoded_line=np.zeros(max_seq_len,np.int64)
                for i,word in enumerate(line[1:]):
                    encoded_line[i]=word2idx[word]                   
                self.encoded_lines.append(encoded_line)
    def __getitem__(self,idx):
        if self.augment:
            indice=np.arange(self.max_seq_len,dtype=np.int32)
            np.random.shuffle(indice)
            input_sentence=self.encoded_lines[idx].copy()
            input_sentence[indice[:20]]=0
            return input_sentence,self.targets[idx]
        else:
            return self.encoded_lines[idx],self.targets[idx]
    
    def __len__(self):
        return len(self.targets)