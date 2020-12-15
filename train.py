import json
import nltk
from nltk_utilis import tokenized_word, stem, bag_of_word
import numpy as np
from model import NeuralNet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intent.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for instant in intents['intents']:
    tag = instant['tag']
    tags += [tag]
    for pattern in instant['patterns']:
        w = tokenized_word(pattern)
        all_words += w
        xy += [(w,tag)]

ignored_words = ['/', '?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignored_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []
for (pattern_sent, tag) in xy:
    bag = bag_of_word(pattern_sent, all_words)
    X_train += [bag]

    label = tags.index(tag)
    Y_train += [label]

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size)

model = NeuralNet(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (w, labels) in train_loader:

        outputs = model(w)
        loss = criterion(outputs, labels.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%100 == 0:
        print(f'epoch = {epoch+1} / {num_epochs}, loss = {loss.item():.4f} ')

print(f'Final loss, loss = {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')