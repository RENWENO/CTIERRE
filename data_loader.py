#! -*- coding: utf-8 -*-
# @Time    : 2024/6/13 11:04
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

    ###The function outputs text numbers that are normal and greater than 1 for the entity label.
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

    # Design a sentence processing function
    def sentence_label(self,Lable, Text):
        # Enter a sentence and tag and extract all [entities, locations, tags]
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

   

    def relate_extract(self,allsub_text,Relation):
        # Enter all [[Entity, location, tag],...,[Entity, location, tag]]
		# Output [[Position, relation, position],...,[position, relation, position]]
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

    #Write a function that selects a given number of lines to compose a new output
    def new_Text_label(self,Text, Label, num):
        newtext = []
        newlabel = []
        for i in num:
            newtext.append(Text[i])
            newlabel.append(Label[i])

        return newlabel, newtext

    #Write a function output adjacency matrix
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
        :param Text:Input tex
        :param Label: Enter the text tag
        :param relation: Input an entity relationship pair
        :param KDgraph: Input knowledge domain knowledge graph
        :return: returns a list [feature, neighborhood adjacency matrix, entity adjacency matrix]
        """
      
        data=[]
        # Sentence and ID
        sentences=[]
        #Domain graph adjacency matrix
        KD_ADJ=[]
        #Reads each line of Text text
        # Graph adjacency matrix
        RELAADJ=[]
        # Tag list
        LABLE=[]
        for li,line in enumerate(Text):
            print(str(li) )
            #Stores the length of each character
            word_len=[]
            #Converts the line string text to a list, which gets -1 because the last character is' /n '
            if line[-1]=='/n':
                words =list(line[:-1])
            else:
                words=list(line)
            # for token in list(line[:-1]):
            #     words.append(self.tokenizer.tokenize(token))
            #Adding CLS and the relations' use 'and' target 'is equivalent to adding three characters to the source text.
            word = ['[CLS]']+[item for item in words]
            for wl in word:
                word_len.append(1)
            """Calculate the position of each character"""
            token_start_idxs = 1 + np.cumsum([0] + word_len[:-1])
            sentences.append((self.tokenizer.convert_tokens_to_ids(word), token_start_idxs))
            """Calculate the domain knowledge adjacency matrix"""
            lens = len(word)
            Adjacent_matrix = np.zeros([lens, lens])
            for h,A in enumerate(word):
                for l,Aj in enumerate(word):
                    if (A in wenben_word) and (Aj in wenben_word):
                        g_i = wenben_word.index(A)
                        g_j = wenben_word.index(Aj)
                        Adjacent_matrix[h][l] = KDgraph[g_i][g_j]
            KD_ADJ.append(Adjacent_matrix)

            """Calculate the adjacency matrix between relations and entities"""
            sen_lab_re = []#Stores text labels and lists of relationships
            Dlable = Label[li]
            sen_lab = self.sentence_label(Dlable, line)
            sen_lab_re.append(sen_lab)
            rela_ext = self.relate_extract(sen_lab,self.Relation)
            sen_lab_re.append(rela_ext)
            Adj, Sparse_Adj = self.Adjacent_matrix(sen_lab_re, line)
            RELAADJ.append((Adj,Sparse_Adj))
        """Calculate text label labels"""
        for ll,lb in enumerate(Label):
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
            1. padding: The data padding of each batch to the same length (the longest data length in the batch)
            2. aligning: Find a label item in each sentence sequence, and align the text with the label
            3. tensor：translate into the tensor
        """
        #Sentence matrix
        sentences = [x[0] for x in batch]
        #print(sentences)
        #Label matrix
        labels = [x[1] for x in batch]
        #print(labels)
        #Neighborhood adjacency matrix
        kd_adj = [x[2] for x in batch]
        # batch length
        #Relational adjacency matrix
        relaadj = [x[3] for x in batch]

        batch_len = len(sentences)

        # compute length of longest sentence in batch
        max_len = max([len(s[0]) for s in sentences])
        max_label_len1 = max([len(s) for s in labels])
        max_label_len = 0
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        """[[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]"""
        batch_label_starts = []
        batch_mask = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_type_id = self.word_pad_idx * np.ones((batch_len, max_len))

        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0]
            batch_mask[j][:cur_len] = 1
            # index that finds labeled data ([CLS] does not count)
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            # Select the appropriate tags to form the list [1,2,3,4,...] And the value in the corresponding position of the label_starts vector becomes 1
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label 
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            if len(batch_labels[j]) > len(batch_labels[j]):
                print(batch_data[j])
            batch_labels[j][:cur_tags_len] = labels[j]
           

        # padding padding the adjacency graph to 0
        batch_Comatrix = np.zeros((batch_len, max_len, max_len))
        for lj in range(batch_len):
            cur_Comat_len = len(kd_adj[lj])
            for hang in range(cur_Comat_len):
                batch_Comatrix[lj][hang][:cur_Comat_len]= kd_adj[lj][hang][:]

        #padding the diagram label to 0
        batch_relamatrix = np.zeros((batch_len, max_len, max_len))
        for lj in range(batch_len):
            cur_Comat_len = len(relaadj[lj][0])
            for hang in range(cur_Comat_len):
                batch_relamatrix[lj][hang][:cur_Comat_len]= relaadj[lj][0][hang][:]
            aa = batch_relamatrix[lj]

        # convert data to torch LongTensors 
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
        L = Lable[:-1].split(" ")
        T = list(Text[:-1])
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

    # relate_extract()

def relate_extract(allsub_text,Relation):
        # Enter all [[Entity, location, tag],...,[Entity, location, tag]]
		# Output [[Position, relation, position],...,[position, relation, position]]
        relate_list = []
        for N in range(len(allsub_text)):
            for M in range(len(allsub_text)):
                for re in Relation:
                    # print(re)
                    if (allsub_text[N][2] in re) and (allsub_text[M][2] in re) and (
                            allsub_text[N][2] != allsub_text[M][2]):
                        each_relate = []
                        relation = re.split(" ")[1]
                        each_relate.append(allsub_text[N][1])
                        each_relate.append(relation)
                        each_relate.append(allsub_text[M][1])
                        relate_list.append(each_relate)

                        
                        each_relate = []
        
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
