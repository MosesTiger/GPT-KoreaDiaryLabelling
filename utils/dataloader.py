import json

import torch
from torch.utils.data import Dataset

class DiaryDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        content = item['content']
        labels = [int(score) for score in item['gpt_response'].strip('[]').split(',')]
        return content, torch.tensor(labels, dtype=torch.float32)

def get_dataloader(data_path, batch_size, shuffle=True):
    
    dataset = DiaryDataset(data_path)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, val_dataloader