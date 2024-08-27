#! -*- coding: utf-8 -*-
# @Time    : 2023/6/30 15:47
# @Author  :
import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from transformers import BertModel
from config import parsers
#from Threa_relation_extart.model.Run import  data_split
from Threa_relation_extart.data.data_loader import NERGRAPH,Stand_Output,data_split
from Threa_relation_extart.data.Domain_Knowledge import  DomainKnowledge
#from Threa_relation_extart.model import Bert_GCN
from torch.utils.data import DataLoader
from torch.optim import AdamW,SparseAdam
from Threa_relation_extart.model.gloablpointer import GlobalPointer
from transformers import BertTokenizer
from  Threa_relation_extart.model.GHM_loss import GHMC_Loss

class NormLayer(nn.Module):
    def __init__(self, num_features, norm_type='batchnorm', eps=1e-5, momentum=0.1):
        super(NormLayer, self).__init__()

        if norm_type == 'batchnorm':
            self.norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        elif norm_type == 'layernorm':
            self.norm = nn.LayerNorm([num_features, 768], eps=eps, elementwise_affine=True)
        elif norm_type == 'instancenorm':
            self.norm = nn.InstanceNorm2d(num_features, eps=eps, momentum=momentum, affine=True)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x):
        return self.norm(x)


class DataFusionModel(nn.Module):
    def __init__(self, num_features, norm_type='layernorm'):
        super(DataFusionModel, self).__init__()
        self.norm_layer1 = NormLayer(num_features, norm_type)
        self.norm_layer2 = NormLayer(num_features, norm_type)
        self.fusion_layer = nn.Linear(2*num_features, num_features)

    def forward(self, x1, x2):

        x1_norm = self.norm_layer1(x1)
        x2_norm = self.norm_layer2(x2)


        fused_features = torch.cat((x1_norm, x2_norm), dim=2)


        output = self.fusion_layer(fused_features)

        return output


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #nn.Parameter()自定义要学习的参数。torch.FloatTensor(in,out)构建一个float类型的大小为(in,out)的张量。
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):#初始化参数
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, text, adj):
        #torch.matmul()两个张量相乘
        hidden = torch.matmul(text.float(), self.weight.float())
        #torch.sum求和
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj.float(), hidden) / denom
        if self.bias is not None:
            output = output + self.bias

        return F.relu(output.type_as(text))

class Bert_GCN(nn.Module):
    def __init__(self):
        super(Bert_GCN,self).__init__()
        self.bert = BertModel.from_pretrained("../Bert/Robert")
        self.gcn1 = GraphConvolution(768,768)
        self.gcn2 = GraphConvolution(768,768)
        self.line = nn.Linear(768,768)
        self.line1 = nn.Linear(768,768)
        self.gatr  = GAT_relation(768)
        self.expand = nn.Linear(1,2)
        self.w = nn.Parameter(torch.ones(2))
        self.gp = GlobalPointer(1,64)
        self.Leakru = nn.LeakyReLU()
        self.NL =  DataFusionModel(768)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,x):
        #x包含的是三个数组
        text, attention_mask, token_type_ids,_,_,adj,_ = x
        outputs = self.bert(input_ids=text, attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            output_hidden_states=True  # 确保 hidden_states 的输出有值
                            )
        outputs = outputs[0]
        g1 = self.gcn1(outputs,adj)
        g1 = self.Leakru(g1)
        #g2 = self.gcn2(g1,adj)
        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        Fout = self.NL(outputs,g1)
        L1   = self.line(Fout)
        L1   = self.dropout(L1)
        L2   = self.Leakru(L1)
        zout = self.gatr(L2,L2)
        FG = self.gp(zout,attention_mask)
        #print(FG)
        return FG
    #Ft维度是（B,L,L,2）


def dot_product_decode(Z,ZT):
	A_pred = torch.sigmoid(torch.matmul(Z,ZT))
	return A_pred

#关系网络注意力机制
class RelationAttention(nn.Module):
    def __init__(self, hidden_size):
        super(RelationAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(2 * hidden_size, 1)
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, p, x, mask=None):
        q = self.query(p)
        k = self.key(x)
        score = self.fuse(q, k)
        if mask is not None:
            mask = 1 - mask[:, None, :].expand(-1, score.size(1), -1)
            score = score.masked_fill(mask == 1, -1e9)
        score = F.softmax(score, 2)
        v = self.value(x)
        out = torch.einsum('bcl,bld->bcd', score, v) + p
        g = self.gate(torch.cat([out, p], 2)).sigmoid()
        out = g * out + (1 - g) * p
        return out

    def fuse(self, x, y):
        x = x.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        y = y.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        temp = torch.cat([x, y], 3)
        return self.score(temp).squeeze(3)

#计算词语与关系之间的注意力
class GAT_relation(nn.Module):
    def __init__(self,hidden_size):
        super(GAT_relation,self).__init__()
        self.ra1 = RelationAttention(hidden_size)
        self.ra2 = RelationAttention(hidden_size)

    def forward(self, x, p, mask=None):
        # 计算字词和关系的注意力
        x_ = self.ra1(x, p)
        x = x_ + x
        # 计算关系和字词的注意力
        p_ = self.ra2(p, x, mask)
        p = p_ + p
        return x+p


def accdef(y_pre,y_true):
    Ft = 2.0 - y_pre
    oneT = Ft.eq(y_true)
    #print(oneT)
    sumont = oneT.float().sum()
    sumTrue = y_true.sum()
    print(sumont)
    print(sumTrue)

    return sumont/sumTrue




if __name__ == '__main__':

    a = [101, 679, 6814, 8024, 517, 2208, 3360, 2208, 1957, 518, 7027, 4638,
         1957, 4028, 1447, 3683, 6772, 4023, 778, 8024, 6844, 1394, 3173, 4495,
         807, 4638, 6225, 830, 5408, 8024, 5445, 3300, 1126, 1767, 3289, 3471,
         4413, 4638, 2767, 738, 976, 4638, 3683, 6772, 1962, 511, 0, 0,
         0, 0, 0]
    t = torch.tensor(a, dtype=torch.long)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    word = tokenizer.convert_ids_to_tokens(t[1].item())
    sent = tokenizer.decode(t.tolist())
    print(word)
    print(sent)

