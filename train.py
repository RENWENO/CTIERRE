#! -*- coding: utf-8 -*-
# @Time    : 2023/6/30 15:47
# @Author  :
import torch
from Threa_relation_extart.model.Run import  data_split
from Threa_relation_extart.data.data_loader import NERGRAPH,Stand_Output,data_split
from Threa_relation_extart.data.Domain_Knowledge import  DomainKnowledge
from Threa_relation_extart.model import Bert_GCN
from torch.utils.data import DataLoader
from torch.optim import AdamW

class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall
def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()

def accdef(y_pre,y_true):
    Ft = 2.0 - y_pre
    oneT = Ft.eq(y_true)
    #print(oneT)
    sumont = oneT.float().sum()
    sumTrue = y_true.sum()
    print(sumont)
    print(sumTrue)

    return sumont/sumTrue


if __name__ == "__main__":
    with open("../data/train_data.txt", "r", encoding="utf-8") as f1:
        TEXT = f1.readlines()

    with open("../data/train_lable.txt", "r", encoding="utf-8") as f2:
        LABLE = f2.readlines()

    with open("../data/test_data.txt", "r", encoding="utf-8") as f1:
        TTEXT = f1.readlines()

    with open("../data/test_lable.txt", "r", encoding="utf-8") as f2:
        TLABLE = f2.readlines()


    with open("../data/relation", "r", encoding="utf-8") as f3:
        relation = f3.readlines()

    gpu = ''
    if gpu != '':
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    Metrics =MetricsCalculator()
    NUM = Stand_Output(LABLE)
    TNUM = Stand_Output(TLABLE)

    train_size = len(NUM)
    Text = []
    Lable = []
    for j,i in enumerate(NUM):
        Text.append(TEXT[i])
        Lable.append(LABLE[i])
        # if j == 31 :
        #     break

    ttrain_size = len(TNUM)
    TText = []
    TLable = []
    for j, i in enumerate(NUM):
        TText.append(TEXT[i])
        TLable.append(LABLE[i])
        # if j == 31:
        #     break
    print("xxxx")
    dk = DomainKnowledge("../data/wenben.txt", "../data/biaoqian.txt")
    M, DT = dk.returner()
    train_dataset = NERGRAPH(Text, Lable, DT, relation, M)
    train_loader = DataLoader(train_dataset, batch_size=8,
                              shuffle=False, collate_fn=train_dataset.collate_fn)
    test_dataset = NERGRAPH(TText, TLable, DT, relation, M)
    test_loader = DataLoader(test_dataset, batch_size=8,
                              shuffle=False, collate_fn=train_dataset.collate_fn)

    model = Bert_GCN()
    model.train()
    model.to(device)
    print(model)
    opt = AdamW(model.parameters(),lr=1e-20)
    #opt = SparseAdam(model.parameters(),lr=1e-6,betas=(0.9, 0.999), eps=1e-08)
    #loss_fn = multilabel_categorical_crossentropy()

    for epoch in range(50):
        loss_sum, count,acc_sum,f1_sum, precision_sum, recall_sum = 0, 0, 0, 0, 0, 0
        for i,batch in enumerate(train_loader):
            torch.set_printoptions(profile="full")
            pre  = model(batch)
            pre1 = pre
            zero = torch.zeros_like(pre1)
            one  = torch.ones_like(pre1)
            pre1 = torch.where(pre1 < 0, zero, pre1)
            pre1 = torch.where(pre1 > 0, one, pre1)
            pre1 = pre1.squeeze(1)
            pre2 = pre.squeeze(1)
            per3 = torch.sigmoid(pre2)
            label = batch[-1]
            loss = multilabel_categorical_crossentropy(label,pre)
            #loss = GHMC_Loss(10,0.5)(per3,label)
            acc  = accdef(pre1,label)
            f1, precision, recall =Metrics.get_evaluate_fpr(pre1,label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss)
            loss_sum = loss_sum + loss
            acc_sum = acc_sum + acc
            f1_sum = f1_sum+f1
            precision_sum = precision_sum+precision
            recall_sum = recall_sum+recall
            count += 1

        LS=loss_sum/count
        ACC= acc_sum/count
        F1 = f1_sum/count
        P = precision_sum/count
        R = recall_sum/count
        print("loss",LS,"acc",ACC,"f1",F1,"precision",P,"recall",R)
        tloss_sum, tcount, tacc_sum, tf1_sum, tprecision_sum, trecall_sum = 0, 0, 0, 0, 0, 0
        for i, batch in enumerate(test_loader):
            torch.set_printoptions(profile="full")
            with torch.no_grad():
                pre = model(batch)
            pre1 = pre
            zero = torch.zeros_like(pre1)
            one = torch.ones_like(pre1)
            pre1 = torch.where(pre1 < 0, zero, pre1)
            pre1 = torch.where(pre1 > 0, one, pre1)
            pre1 = pre1.squeeze(1)
            pre2 = pre.squeeze(1)
            per3 = torch.sigmoid(pre2)
            label = batch[-1]
            tloss = multilabel_categorical_crossentropy(label, pre)
            # loss = GHMC_Loss(10,0.5)(per3,label)
            tacc = accdef(pre1, label)
            tf1, tprecision, trecall = Metrics.get_evaluate_fpr(pre1, label)
            print(loss)
            tloss_sum = tloss_sum + loss
            tacc_sum = tacc_sum + tacc
            tf1_sum = tf1_sum + tf1
            tprecision_sum = tprecision_sum + tprecision
            trecall_sum = trecall_sum + trecall
            tcount += 1

        TLS = tloss_sum / tcount
        TACC = tacc_sum / tcount
        TF1 = tf1_sum / tcount
        TP = tprecision_sum / tcount
        TR = trecall_sum / tcount
        print("test_loss", TLS, "test_acc", TACC, "test_f1", TF1, "test_precision", TP, "Test_recall", TR)



