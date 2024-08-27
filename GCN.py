#! -*- coding: utf-8 -*-
# @Time    : 2023/8/3 13:26
# @Author  : LiuGan
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight) # 权重self.weight随机产生
        denom = torch.sum(adj, dim=2, keepdim=True) + 1  # 加一保证做除法时分母不为零
        output = torch.matmul(adj, hidden) / denom
        output = F.relu(output + self.bias)
        #print(output)
        return output
def main():
    # 假设该句子经过构建依赖树后的邻接矩阵为adj
    adj =torch.tensor([
        [   [1., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
    ],
        [
            [1., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
        ]
    ])
    # 假设一个句子中有10个单词，从前向后单词对应的索引为[0, 1, 2, 3, 3, 4, 6,0, 1, 2]
    input = torch.tensor([[0, 1, 2, 3, 3, 4, 6, 0, 1, 2],
                          [1, 2, 4, 6, 7, 3, 5, 7, 8, 9]], dtype=torch.long)
    embedding = torch.nn.Embedding(10, 50)
    x = embedding(input)  # 生成每个单词对应的词嵌入，维度为50
    print(x)
    gc1 = GraphConvolution(50, 30)
    gc2 = GraphConvolution(30,6)
    x1=gc1(x, adj)
    x2=gc2(x1,adj)

    print(x2)
if __name__ == '__main__':
    main()
