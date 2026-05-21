import os
import pickle

import torch
import numpy as np


class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, flag, PAD_TRUNC_DIGIT):
        pkl_path = f'./dataset_localization/{dataset}/{flag}.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            self.traffic_seq = torch.tensor(data['pkt'], dtype=torch.long)
            self.header_seq = torch.tensor(data['head'], dtype=torch.long)
            self.label = torch.tensor(data['label'], dtype=torch.long)
            self.PAD_TRUNC_DIGIT = PAD_TRUNC_DIGIT
            return

        if flag == 'train':
            self.data = np.load(f'./dataset/{dataset}/train_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_train_pyg.npz')
        elif flag == 'val':
            self.data = np.load(f'./dataset/{dataset}/val_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_val_pyg.npz')
        else:
            self.data = np.load(f'./dataset/{dataset}/test_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_test_pyg.npz')

        seq, label = self.data['data'], self.data['label']
        header_seq = self.header['data']
        
        self.traffic_seq = torch.tensor(seq)
        self.header_seq = torch.tensor(header_seq)
        self.label = torch.tensor(label)


        self.PAD_TRUNC_DIGIT = PAD_TRUNC_DIGIT

    def __getitem__(self, idx):
        return self.traffic_seq[idx], self.label[idx], (self.traffic_seq[idx] == self.PAD_TRUNC_DIGIT).long(), self.header_seq[idx], (self.header_seq[idx] == self.PAD_TRUNC_DIGIT).long()


    def __len__(self):
        return len(self.traffic_seq)


class TrafficDatasetClass(torch.utils.data.Dataset):
    def __init__(self, dataset, flag, class_idx, PAD_TRUNC_DIGIT, pred_label=None):
        if flag == 'train':
            self.data = np.load(f'./dataset/{dataset}/train_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_train_pyg.npz')
        elif flag == 'val':
            self.data = np.load(f'./dataset/{dataset}/val_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_val_pyg.npz')
        else:
            self.data = np.load(f'./dataset/{dataset}/test_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_test_pyg.npz')

        seq, label = self.data['data'], self.data['label']
        header_seq = self.header['data']

        seq = seq[label == class_idx]
        header_seq = header_seq[label == class_idx]
        label = label[label == class_idx]

        self.traffic_seq = torch.tensor(seq)
        self.header_seq = torch.tensor(header_seq)
        self.label = torch.tensor(label)
        self.pred_label = pred_label
        self.PAD_TRUNC_DIGIT = PAD_TRUNC_DIGIT
        if pred_label is not None:
            self.pred_labels = torch.tensor(pred_label)

    def __getitem__(self, idx):
        item = (
            self.traffic_seq[idx],
            self.label[idx],
            (self.traffic_seq[idx] == self.PAD_TRUNC_DIGIT).long(),
            self.header_seq[idx],
            (self.header_seq[idx] == self.PAD_TRUNC_DIGIT).long(),
        )
        if self.pred_label is not None:
            return item + (self.pred_labels[idx],)
        return item

    def __len__(self):
        return len(self.traffic_seq)
