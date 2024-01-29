import torch
from torch.utils.data import Dataset

class DblpmDataset(Dataset):
    def __init__(self, pos_path, neg_path, transform=None, target_transform=None):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.pos_f = open(self.pos_path, 'r').readlines()
        self.neg_f = open(self.neg_path, 'r').readlines()
        self.transform = transform
        self.target_transform = target_transform
        self.pos_lis = []
        self.neg_lis = []
        for line in self.pos_f:
            self.pos_lis.append(list(map( lambda x: int(x), line.strip().split('\t'))))
        for line1 in self.neg_f:
            self.neg_lis.append(list(map( lambda x: int(x), line1.strip().split(' '))))

    def __len__(self):
        return len(self.pos_f)

    def __getitem__(self, idx):
        # print(idx,'----:')
        pos = torch.tensor(self.pos_lis[idx])
        neg = torch.tensor(self.neg_lis[idx])
        if self.transform:
            pass
        if self.target_transform:
            pass
        return pos, neg


class DblpmDatasetTest(Dataset):
    def __init__(self, pos_path, neg_path,transform=None, target_transform=None):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.pos_f = open(self.pos_path, 'r').readlines()
        self.neg_f = open(self.neg_path, 'r').readlines()
        self.transform = transform
        self.target_transform = target_transform
        self.pos_lis = []
        self.neg_lis = []
        for line in self.pos_f:
            # print(line)
            li = list(map( lambda x: int(x), line.strip().split('\t')))
            li.append(1)
            # print('test',li)
            self.pos_lis.append(li)
        for line1 in self.neg_f:
            li1 = list(map( lambda x: int(x), line1.strip().split('\t')))
            li1.append(0)
            self.neg_lis.append(li1)

    def __len__(self):
        return len(self.pos_f)

    def __getitem__(self, idx):
        # print(idx,'----:')

        pos = torch.tensor(self.pos_lis[idx])
        neg = torch.tensor(self.neg_lis[idx])
        if self.transform:
            pass
        if self.target_transform:
            pass
        return pos,neg


class RateDataset(Dataset):
    def __init__(self, pos_path,  transform=None, target_transform=None):
        self.pos_path = pos_path

        self.pos_f = open(self.pos_path, 'r').readlines()

        self.transform = transform
        self.target_transform = target_transform
        self.pos_lis = []

        for line in self.pos_f:
            self.pos_lis.append(list(map( lambda x: float(x), line.strip().split(' '))))


    def __len__(self):
        return len(self.pos_f)

    def __getitem__(self, idx):
        # print(idx,'----:')
        pos = torch.tensor(self.pos_lis[idx])

        if self.transform:
            pass
        if self.target_transform:
            pass
        return pos
