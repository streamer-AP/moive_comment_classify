import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Text_Dataset
from model import TextClassify
from util import Average

if __name__ == "__main__":
    with open("config.json","r") as f:
        cfg=json.load(f)
    with open(cfg["word2idx"], "r") as f:
        word2idx = json.load(f)

    net = TextClassify(cfg["vocab_size"], cfg["max_seq_len"],
                       cfg["embedding_size"], nums_class=2)
    net.cuda()
    net.load_state_dict(torch.load("best_model.pth"))
    acc_cnt=0
    cnt=0

    with open(cfg["test_txt_path"],"r") as f:
        lines=f.readlines()
        for line in lines:
            line=line.replace("\t"," ")
            line=line.strip().strip("\n").split(" ")
            if len(line)<2:
                continue
            target=eval(line[0])
            encoded_line=np.zeros((1,cfg["max_seq_len"]),np.int64)
            for i,word in enumerate(line[1:]):
                encoded_line[0][i]=word2idx[word]
            encoded_line=torch.from_numpy(encoded_line).cuda()
            predict=torch.argmax(net(encoded_line[:2]))
            print(f"sentence: {''.join(line[1:])},\n target emotion {target}, predict emotion {predict}")
            if target==predict:
                acc_cnt+=1
            cnt+=1
    print(f"predict {cnt} sentence, correct {acc_cnt} sentence, accuracy {acc_cnt/float(cnt)}")
        

