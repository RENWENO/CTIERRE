#! -*- coding: utf-8 -*-
# @Time    : 2024/6/21 16:47
# @Author  :
import collections
import numpy as np


class DomainKnowledge(object):
    def __init__(self,textdir,labledir):
        self.textdir = textdir
        self.labledir = labledir
        self.listtext,self.listlable = self._read(self.textdir,self.labledir)
        self.DomainMatrix,self.DomainText = self._WordFrequencyStatistics(self.listtext,self.listlable)
    def _read(self,textdir,labledir):
        with open(labledir, 'r', encoding='utf-8') as f:
            biaoqian = f.readlines()
        with open(textdir, 'r', encoding='utf-8') as f1:
            wenben = f1.readlines()
        return  wenben,biaoqian

    def pmi(self,M, positive=True):
        col_totals = M.sum(axis=0)
        row_totals = M.sum(axis=1)
        expected = np.outer(col_totals, row_totals) / M.sum()
        with np.errstate(divide="ignore"):
            res_log = np.log(M / (expected + 1e-10))
        res_log[np.isinf(res_log)] = 0.0
        if positive:
            res_log[res_log < 0] = 0.0
        return res_log

    def _WordFrequencyStatistics(self,listtext,listlable):
        WBList = []
        # Convert text to a list
        for wb in listtext:
            list = []
            for i in wb:
                if i == "\n":
                    continue
                list.append(i)
            WBList.append(list)

        # print(WBList[1])
        # print(biaoqian[1])
        # print(len(WBList[1]))
        # print(len(biaoqian[1].split(" ")))
        allThrea = []
        Threa_WB = []
        word = []
        # Entities that collect cyber threat intelligence
        for num in range(len(listlable)):
            for iter, label in enumerate(listlable[num].split(" ")):
                word.append(WBList[num][iter])
                if label != "O":
                    if label[0] == "B":
                        Threa_WB = []
                        Threa_WB.append(WBList[num][iter])
                    else:
                        Threa_WB.append(WBList[num][iter])
                if label[0] == "E":
                    allThrea.append(Threa_WB)

        #print(allThrea)

        allpairsword = []
        # Statistical occurrence
        word_counts = collections.Counter(word)
        leng = len(word_counts)
        concurrence_matrix = np.zeros([leng, leng])
        wenben_word = []
        for wc in word_counts:
            wenben_word.append(wc[0])
            # print(wc)

        # Collect adjacent string
        for line in allThrea:
            for i in range(0, len(line)):
                for j in range(0, len(line)):
                    string = line[i] + line[j]
                    allpairsword.append(string)

        #print(len(allpairsword))
        for pairsword in allpairsword:
            Start = pairsword[0]
            End = pairsword[1]
            i = wenben_word.index(Start)
            j = wenben_word.index(End)
            concurrence_matrix[i][j] = concurrence_matrix[i][j] + 1
        #print(concurrence_matrix.sum())
        M_pmi = self.pmi(concurrence_matrix)
        np.set_printoptions(precision=2)
        M = np.nan_to_num(M_pmi)
        return M,wenben_word
    def returner(self):
        return self.DomainMatrix,self.DomainText

if __name__ == '__main__':
    dk = DomainKnowledge("wenben.txt","biaoqian.txt")
    M,DT = dk.returner()
    print(type(M))
    print(len(DT))




