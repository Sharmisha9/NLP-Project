'''
Complete Training Script for Joke Generation Model
'''

# Preliminaries
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')
import config

# Initializing model and adding the PAD token
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
special_tokens_dict = {'pad_token': '<PAD>'}
num_added_toks = config.Tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(config.Tokenizer))
print(f'We have added {num_added_toks} tokens')

# Dataset Class
class JokesDataset(Dataset):
    '''
    Custom dataset for DataLoader
    '''
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.eos_tok = "<EOS>"
        self.data['Joke'] = self.data['Joke'].apply(lambda x: "JOKE: " + str(x) + self.eos_tok)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        joke = self.data.iloc[idx, 1]
        inputs = self.tokenizer.encode_plus(
            joke,
            None,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            padding='max_length',
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(ids, dtype=torch.long)
        }

# Training Function
def train_fn(data_loader, model, optimizer, device, scheduler, epoch):
    model.train()
    for bi, d in enumerate(data_loader):
        ids = d["ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        targets = d['targets'].to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (bi+1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{config.EPOCHS}], Step [{bi+1}/{len(data_loader)}], Loss: {loss.item():.4f}')

# Engine
def run():
    jokes = pd.read_csv(config.TRAIN_PATH)
    dataset = JokesDataset(jokes, config.Tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_train_steps = int(len(dataloader) / config.BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    for epoch in range(config.EPOCHS):
        print(f"EPOCH {epoch+1} started" + '=' * 30)
        train_fn(dataloader, model, optimizer, device, scheduler, epoch)

        models_folder = config.MODEL_FOLDER
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_joke_generator_epoch{epoch}.pt"))

if __name__ == "__main__":
    run()
