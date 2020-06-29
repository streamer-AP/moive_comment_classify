import torch
from torch import nn

class TextClassify(nn.Module):
    def __init__(self,vocabulary_size,max_seq_len=700,embedding_size=200,nums_class=2):
        super().__init__()
        self.embedding=nn.Embedding(vocabulary_size,embedding_size)
        self.lstm=nn.LSTM(embedding_size,128,2,batch_first=True,dropout=0.5)
        self.conv1=nn.Conv1d(max_seq_len,64,1)
        self.conv2=nn.Conv1d(64,1,1)

        self.classify=nn.Linear(64,2)
    def forward(self,x):
        x=self.embedding(x)
        x,h=self.lstm(x)
        x=self.conv1(x)
        x=nn.functional.relu(x)
        x=self.conv2(x)
        x=nn.functional.relu(x)
        x=torch.squeeze(x)
        x=nn.functional.relu(x)
        return x
