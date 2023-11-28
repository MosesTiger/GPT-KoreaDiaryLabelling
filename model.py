import json
import torch
from transformers import BigBirdTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import BigBirdForSequenceClassification
from torch.optim import AdamW

original_file_path = "C:\\Users\\PC\\Downloads\\your_new_json_file.json"

class DiaryDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        content = item['content']
        labels = [int(score) for score in item['gpt_response'].strip('[]').split(',')]
        encoding = self.tokenizer(content, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        return {**encoding, 'labels': torch.tensor(labels, dtype=torch.long)}


with open(original_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
dataset = DiaryDataset(data, tokenizer)
train_dataloader = DataLoader(dataset, shuffle=True, batch_size=8)
model = BigBirdForSequenceClassification.from_pretrained("google/bigbird-roberta-base", num_labels=5)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3  

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].squeeze()
        attention_mask = batch['attention_mask'].squeeze()
        labels = batch['labels']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")
    
model.eval()

torch.save(model.state_dict(), "model.pth")


