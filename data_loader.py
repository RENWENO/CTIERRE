#! -*- coding: utf-8 -*-
# @Time    : 2023/7/13 11:04
# @Author  :
import random

import torch

import numpy as np
from scipy.sparse import coo_matrix
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from scipy.sparse import
from Threa_relation_extart.data.Domain_Knowledge import  DomainKnowledge


class NERGRAPH(Dataset):
    def __init__(self,Text,Label,wenben_word,relation,KDgraph, word_pad_idx=0, label_pad_idx=-1):

        self.tokenizer = BertTokenizer.from_pretrained("../Bert/Robert", do_lower_case=True)
        #self.label2id = config.label2id
        #self.id2label = {_id: _label for _label, _id in list(config.label2id.items())}

        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = torch.device("cpu")
        self.Relation = relation
        self.dataset = self.preprocess(Text, Label, KDgraph, wenben_word)

    ###写函数输出实体标签正常且数量大于1的文本编号
    def Stand_Output(self,Lable):
        TRUER = 1
        num = []
        for si, sencent in enumerate(Lable):
            Ts = []
            S = sencent.split("O")
            # S.remove(' ')
            for t in S:
                if (t == ' ') or (t == '') or (t == '\n'):
                    continue
                else:
                    Ts.append(t)
            for ts in Ts:
                # print(ts)
                ts = ts[1:-1]
                NER = ts.split(' ')
                l = NER[0][2:]
                for li in range(1, len(NER) - 1):
                    if 'I_' + l != NER[li]:
                        TRUER = 0
                if ('B_' + l != NER[0]) and ('E_' + l != NER[-1]):
                    TRUER = 0

            if (TRUER == 1) and len(Ts) > 1:
                num.append(si)
            else:
                TRUER = 1
        # print(Ts)
        # print(num)
        # print(len(num))
        return num

    # 设计一个句子处理函数
    def sentence_label(self,Lable, Text):
        # 输入一个句子和标签，提取出所有的[实体、位置、标签]
        if Lable[-1] == "/n":
            L = Lable[:-1].split(" ")
        else:
            L =  Lable[:].split(" ")
        if Text[-1] == "/n":
            T = list(Text[:-1])
        else:
            T = list(Text)
        # print(Lable[0])
        # print(Text[0])
        # print(len(Relation))
        # print(L,len(L))
        # print(T,len(T))
        alllocation = []
        allsub_text = []
        for si, chat in enumerate(L):
            if len(chat) > 1:
                if chat[0] == "B":
                    location = []
                    sub_text = []
                    location.append(si)
                    sub_text.append(T[si])

                elif chat[0] == "E":
                    lab2text = []
                    location.append(si)
                    sub_text.append(T[si])
                    string = "".join(sub_text)
                    lab = chat[2:]
                    # loc = " ".join(location)
                    lab2text.append(string)
                    lab2text.append(location)
                    lab2text.append(lab)
                    alllocation.append(location)
                    allsub_text.append(lab2text)
                    location = []
                    sub_text = []
                elif chat[0] == "I":
                    sub_text.append(T[si])

        # print(allsub_text)
        return allsub_text

    # print(alllocation)

    # 关系抽取函数relate_extract()

    def relate_extract(self,allsub_text,Relation):
        # 输入所有的[[实体、位置、标签],...,[实体、位置、标签]]
        # 输出[[位置，关系，位置],...,[位置，关系，位置]]
        relate_list = []
        for N in range(len(allsub_text)):
            for M in range(len(allsub_text)):
                for re in Relation:
                    # print(re)
                    if (allsub_text[N][2] in re) and (allsub_text[M][2] in re) and (
                            allsub_text[N][2] != allsub_text[M][2]):
                        each_relate = []
                        relation = re.split(" ")[1]
                        # print(allsub_text[N][2])
                        # print(allsub_text[M][2])
                        # print(relation)
                        each_relate.append(allsub_text[N][1])
                        each_relate.append(relation)
                        each_relate.append(allsub_text[M][1])
                        relate_list.append(each_relate)
                        each_relate = []
        return relate_list

    # 写一个函数对给出的特定行数进行挑选，组成新的输出
    def new_Text_label(self,Text, Label, num):
        newtext = []
        newlabel = []
        for i in num:
            newtext.append(Text[i])
            newlabel.append(Label[i])

        return newlabel, newtext

    # 写一个函数输出邻接矩阵
    def Adjacent_matrix(self,List, Text):
        leng = len(Text[:-1])
        # print(leng)
        Axj = np.zeros([2,leng, leng])
        # print(Axj)
        for re in List[1]:
            re[0]
            # print(type(re[0][0]))

            if re[1] == "uses":
                Sh = re[0][0]
                Sl = re[0][1]
                Oh = re[2][0]
                Ol = re[2][1]
                Axj[0][Sh][Sl] = 1.0
                Axj[0][Oh][Ol] = 1.0
                if Sl < Oh:
                    Axj[0][Sl][Oh] = 1.0
                else:
                    Axj[0][Oh][Sl] = 1.0
            if re[1] == "targets":
                Sh = re[0][0]
                Sl = re[0][1]
                Oh = re[2][0]
                Ol = re[2][1]
                Axj[1][Sh][Sl] = 1.0
                Axj[1][Oh][Ol] = 1.0
                if Sl < Oh:
                    Axj[1][Sl][Oh] = 1.0
                else:
                    Axj[1][Oh][Sl] = 1.0

        # print(Axj)
        coo_np = coo_matrix(Axj)
        # data = coo_np.data
        # print(coo_np)
        return Axj, coo_np

    def preprocess(self,Text,Label,KDgraph,wenben_word):
        """
        :param Text: 输入的文本
        :param Label: 输入的文本标签
        :param relation: 输入的实体关系对
        :param KDgraph: 输入知识领域知识图
        :return: 返回一个列表[特征，邻域关系邻接关系矩阵，实体关系邻接矩阵]
        """
        #Text = Text[:-1] 每一行的最后一列是“/n”
        #Label = Label[:-1] 每一行的最后一列是“/n”
        data=[]
        #句子和ID
        sentences=[]
        #领域图邻接矩阵
        KD_ADJ=[]
        #读取Text文本中的每一行
        #关系图邻接矩阵
        RELAADJ=[]
        #标签列表
        LABLE=[]
        for li,line in enumerate(Text):
            print("计算第"+ str(li) +"条文本和图构建")
            #存储每个字符的长度
            word_len=[]
            #将line字符串文本转化成列表，由于最后一个字符为‘/n’所以取到-1
            if line[-1]=='/n':
                words =list(line[:-1])
            else:
                words=list(line)
            # for token in list(line[:-1]):
            #     words.append(self.tokenizer.tokenize(token))
            #加上CLS和关系‘use’和‘target’,相当于对源文本添加了3个字符。
            word = ['[CLS]']+[item for item in words]
            for wl in word:
                word_len.append(1)
            """计算每个字符的位置"""
            token_start_idxs = 1 + np.cumsum([0] + word_len[:-1])
            #将字符转化成ID，并同token_start_idxs位置信息，一起放入到sentences列表中
            sentences.append((self.tokenizer.convert_tokens_to_ids(word), token_start_idxs))
            """计算领域知识邻接矩阵"""
            lens = len(word)
            Adjacent_matrix = np.zeros([lens, lens])
            for h,A in enumerate(word):
                for l,Aj in enumerate(word):
                    if (A in wenben_word) and (Aj in wenben_word):
                        g_i = wenben_word.index(A)
                        g_j = wenben_word.index(Aj)
                        Adjacent_matrix[h][l] = KDgraph[g_i][g_j]
            KD_ADJ.append(Adjacent_matrix)

            """计算出关系和实体之间的邻接矩阵"""
            sen_lab_re = []#存储文本标签和关系列表
            # 输入一个句子和标签，提取出所有的[实体、位置、标签]
            Dlable = Label[li]
            sen_lab = self.sentence_label(Dlable, line)
            sen_lab_re.append(sen_lab)
            # 输入所有的[[实体、位置、标签],...,[实体、位置、标签]]
            # 输出[[位置，关系，位置],...,[位置，关系，位置]]
            rela_ext = self.relate_extract(sen_lab,self.Relation)
            sen_lab_re.append(rela_ext)
            Adj, Sparse_Adj = self.Adjacent_matrix(sen_lab_re, line)
            RELAADJ.append((Adj,Sparse_Adj))
        """计算文本标签标签"""
        for ll,lb in enumerate(Label):
            #print("计算第" + str(ll) + "条文本标签")
            if lb[-1] == "/n":
                Tjlabel=lb[:-1]
            else:
                Tjlabel=lb
            Participle = Tjlabel.split(" ") +["O"]+["O"]
            lblist=[]
            for tb in Participle:
                if tb == "O":
                    lblist.append(0)
                else:
                    lblist.append(1)
            LABLE.append(lblist)

        for sentence, label,kd_adj,relaadj in zip(sentences,LABLE,KD_ADJ,RELAADJ):
            data.append((sentence, label,kd_adj,relaadj))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        kd_adj = self.dataset[idx][2]
        relaadj = self.dataset[idx][3]
        return [word, label, kd_adj, relaadj]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        process batch data, including:
            1. padding: 将每个batch的data padding到同一长度（batch中最长的data长度）
            2. aligning: 找到每个sentence sequence里面有label项，文本与label对齐
            3. tensor：转化为tensor
        """
        #句子矩阵
        sentences = [x[0] for x in batch]
        #print(sentences)
        #标签矩阵
        labels = [x[1] for x in batch]
        #print(labels)
        #邻域邻接矩阵
        kd_adj = [x[2] for x in batch]
        # batch length
        #关系邻接矩阵
        relaadj = [x[3] for x in batch]

        batch_len = len(sentences)

        # compute length of longest sentence in batch
        max_len = max([len(s[0]) for s in sentences])
        max_label_len1 = max([len(s) for s in labels])
        #print(max_len)
        max_label_len = 0

        # padding data 初始化 word_pad_idx=0, label_pad_idx=-1
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        """[[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]"""
        batch_label_starts = []
        batch_mask = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_type_id = self.word_pad_idx * np.ones((batch_len, max_len))

        # padding and aligning补充和对齐
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            #将batch_data中前cur_len的的位置赋值为sentences[j][0]
            batch_data[j][:cur_len] = sentences[j][0]
            batch_mask[j][:cur_len] = 1
            # 找到有标签的数据的index（[CLS]不算）
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            #选择相应的标签组成列表[1，2，3，4，...],并在对应的label_starts向量位置中的值变为1
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)#将文本位置信息添加到batch_label_starts列表中
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label 填充标签将扩充的标签弄成-1
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            #print(len(labels[j]))
            #print(len(batch_labels[j]))
            if len(batch_labels[j]) > len(batch_labels[j]):
                print(batch_data[j])
            batch_labels[j][:cur_tags_len] = labels[j]
            #print("第"+str(j)+"次")

        # 对邻接图进行padding 填充为0
        batch_Comatrix = np.zeros((batch_len, max_len, max_len))
        for lj in range(batch_len):
            cur_Comat_len = len(kd_adj[lj])
            for hang in range(cur_Comat_len):
                batch_Comatrix[lj][hang][:cur_Comat_len]= kd_adj[lj][hang][:]

        #对关系图标签进行padding 填充为0
        batch_relamatrix = np.zeros((batch_len, max_len, max_len))
        for lj in range(batch_len):
            cur_Comat_len = len(relaadj[lj][0])
            for hang in range(cur_Comat_len):
                #print(relaadj[lj][hang][:])
                batch_relamatrix[lj][hang][:cur_Comat_len]= relaadj[lj][0][hang][:]
            aa = batch_relamatrix[lj]

        # convert data to torch LongTensors 将数据转化成torch类型的LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_mask = torch.tensor(batch_mask, dtype=torch.long)
        batch_type_id = torch.tensor(batch_type_id, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        batch_kd_adj = torch.tensor(batch_Comatrix,dtype=torch.long)
        batch_relaadj = torch.tensor(batch_relamatrix,dtype=torch.long)

        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        batch_mask, batch_type_id = batch_mask.to(self.device), batch_type_id.to(self.device)
        batch_labels = batch_labels.to(self.device)
        batch_kd_adj = batch_kd_adj.to(self.device)
        batch_relaadj = batch_relaadj.to(self.device)
        return [batch_data,batch_mask, batch_type_id, batch_label_starts, batch_labels,batch_kd_adj,batch_relaadj]


def Stand_Output(Lable):
        TRUER = 1
        num = []
        for si, sencent in enumerate(Lable):
            Ts = []
            S = sencent.split("O")
            # S.remove(' ')
            for t in S:
                if (t == ' ') or (t == '') or (t == '\n'):
                    continue
                else:
                    Ts.append(t)
            for ts in Ts:
                # print(ts)
                ts = ts[1:-1]
                NER = ts.split(' ')
                l = NER[0][2:]
                for li in range(1, len(NER) - 1):
                    if 'I_' + l != NER[li]:
                        TRUER = 0
                if ('B_' + l != NER[0]) and ('E_' + l != NER[-1]):
                    TRUER = 0

            if (TRUER == 1) and len(Ts) > 1:
                num.append(si)
            else:
                TRUER = 1
        # print(Ts)
        # print(num)
        # print(len(num))
        return num
def martIfzero(text,lable):
    def sentence_label(Lable, Text):
        # 输入一个句子和标签，提取出所有的[实体、位置、标签]
        L = Lable[:-1].split(" ")
        T = list(Text[:-1])
        # print(Lable[0])
        # print(Text[0])
        # print(len(Relation))
        # print(L,len(L))
        # print(T,len(T))
        alllocation = []
        allsub_text = []
        for si, chat in enumerate(L):
            if len(chat) > 1:
                if chat[0] == "B":
                    location = []
                    sub_text = []
                    location.append(si)
                    sub_text.append(T[si])

                elif chat[0] == "E":
                    lab2text = []
                    location.append(si)
                    sub_text.append(T[si])
                    string = "".join(sub_text)
                    lab = chat[2:]
                    # loc = " ".join(location)
                    lab2text.append(string)
                    lab2text.append(location)
                    lab2text.append(lab)
                    alllocation.append(location)
                    allsub_text.append(lab2text)
                    location = []
                    sub_text = []
                elif chat[0] == "I":
                    sub_text.append(T[si])

        # print(allsub_text)
        return allsub_text

    # print(alllocation)

    # 关系抽取函数relate_extract()

def relate_extract(allsub_text,Relation):
        # 输入所有的[[实体、位置、标签],...,[实体、位置、标签]]
        # 输出[[位置，关系，位置],...,[位置，关系，位置]]
        relate_list = []
        for N in range(len(allsub_text)):
            for M in range(len(allsub_text)):
                for re in Relation:
                    # print(re)
                    if (allsub_text[N][2] in re) and (allsub_text[M][2] in re) and (
                            allsub_text[N][2] != allsub_text[M][2]):
                        each_relate = []
                        relation = re.split(" ")[1]
                        # print(allsub_text[N][2])
                        # print(allsub_text[M][2])
                        # print(relation)
                        each_relate.append(allsub_text[N][1])
                        each_relate.append(relation)
                        each_relate.append(allsub_text[M][1])
                        relate_list.append(each_relate)

                        # print(len(relate_list))
                        each_relate = []
        # print(relate_list)
        return relate_list


def Data_split(full_list,ratio):

    list = full_list
    random.shuffle(list)
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list


    sublist_1 = list[:offset]
    sublist_2 = list[offset:]
    return sublist_1, sublist_2

if __name__ == '__main__':
    #pass
    with open("wenben.txt","r",encoding="utf-8") as f1:
        TEXT = f1.readlines()
    with open("biaoqian.txt","r",encoding="utf-8") as f2:
        LABLE = f2.readlines()

    with open("relation","r",encoding="utf-8") as f3:
        relation = f3.readlines()

    NUM = Stand_Output(LABLE)
    print(NUM)
    train,test = Data_split(NUM,ratio=0.2)
    Text=[]
    Lable=[]
    for j,i in enumerate(NUM):
        Text.append(TEXT[i])
        Lable.append(LABLE[i])
        # if j == 16:
        #     break

    print("xxxx")
    dk = DomainKnowledge("wenben.txt", "biaoqian.txt")
    M, DT = dk.returner()
    train_dataset = NERGRAPH(Text,Lable,DT,relation,M)
    train_loader = DataLoader(train_dataset, batch_size=4,
                              shuffle=False, collate_fn=train_dataset.collate_fn)
    for batch in train_loader:
        torch.set_printoptions(profile="full")
        token,mask,type_id,postion,lable,kdgraph,relationgraph = batch
        print(token)
        # print(mask)
        # print(type_id)
        # print(postion)
        # pri