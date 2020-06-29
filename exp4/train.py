import json
import logging
import os
import shutil

import torch
from torch import nn
from torch.optim import Adam
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from dataset import Text_Dataset
from model import TextClassify
from util import Average


def train(cfg, net, train_dl, eval_dl):
    patient = 0
    Loss = nn.CrossEntropyLoss()
    train_loss_avg = Average()
    train_acc_avg = Average()
    eval_loss_avg = Average()
    eval_acc_avg = Average()
    best_model_acc = 0
    if os.path.exists(cfg["log_dir"]):
        shutil.rmtree(cfg["log_dir"])
    writer = tensorboard.SummaryWriter(cfg["log_dir"])
    optim = Adam(net.parameters(), lr=cfg["lr"])
    for epoch in range(cfg["epoch"]):
        step = 0
        train_acc_avg.reset()
        train_loss_avg.reset()
        eval_loss_avg.reset()
        eval_acc_avg.reset()
        for text, target in train_dl:
            optim.zero_grad()
            step += 1
            text = text.cuda()
            target = target.cuda()
            predict = net(text)
            loss = Loss(predict, target)
            loss.backward()
            optim.step()
            train_loss_avg.update(loss.item(), len(target))
            acc_cnt = torch.sum(torch.argmax(predict, dim=1) == target).item()
            train_acc_avg.update(acc_cnt/float(len(target)))
            if step % cfg["display_step"] == 0:
                print(
                    f"epoch {epoch}, step {step}, train acc {train_acc_avg.avg()}, train loss {train_loss_avg.avg()}")
        writer.add_scalar("train_loss", train_loss_avg.avg(), epoch)
        writer.add_scalar("train_acc", train_acc_avg.avg(), epoch)
        acc_cnt=0
        for text, target in eval_dl:
            text = text.cuda()
            target = target.cuda()
            predict = net(text)
            loss = Loss(predict, target)
            eval_loss_avg.update(loss.item(), len(target))
            acc_cnt = torch.sum(torch.argmax(predict, dim=1) == target).item()
            eval_acc_avg.update(acc_cnt/float(len(target)))
        print(
            f"epoch {epoch}, eval acc {eval_acc_avg.avg()}, eval loss {eval_loss_avg.avg()}")

        writer.add_scalar("eval_loss", eval_loss_avg.avg(), epoch)
        writer.add_scalar("eval_acc", eval_acc_avg.avg(), epoch)

        if eval_acc_avg.avg() > best_model_acc:
            best_model_acc = eval_acc_avg.avg()
            torch.save(net.state_dict(), "best_model.pth")
            patient = 0
        else:
            patient += 1
        if patient > cfg["patient_epoch"]:
            break


if __name__ == "__main__":
    with open("config.json", "r") as f:
        cfg = json.load(f)
    with open(cfg["word2idx"], "r") as f:
        word2idx = json.load(f)

    net = TextClassify(cfg["vocab_size"], cfg["max_seq_len"],
                       cfg["embedding_size"], nums_class=2)
    net.cuda()
    train_ds = Text_Dataset(cfg["train_txt_path"],
                            word2idx, cfg["max_seq_len"],augment=True)
    eval_ds = Text_Dataset(cfg["eval_txt_path"], word2idx, cfg["max_seq_len"])
    train_dl = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=cfg["batch_size"], shuffle=False)
    train(cfg, net, train_dl, eval_dl)
