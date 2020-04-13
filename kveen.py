import os
import sys
import random
import requests

import nltk
import bs4 as bs

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import datetime as dt

from torch.autograd import Variable

import string
import time
import math
import unidecode

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--based', type=str, default="")
args = argparser.parse_args()

# .dat sounds cooler and more hackerman than .txt
DATA_PATH = "speeches.dat"

def queen_spider():
    whole = ""

    links = []

    r = requests.get("http://kongehuset.dk/monarkiet-i-danmark/nytarstaler/hendes-majestat-dronningens-nytarstaler")
    soup = bs.BeautifulSoup(r.text, features="lxml")

    links = soup.findAll('div', { 'class': 'field-items' })[4]

    speech_links = []

    for div in links:
        soup2 = bs.BeautifulSoup(str(div), features="lxml")
        
        a = soup2.findAll('a', href=True)

        for a in a:
            speech_links.append(a['href'])

    i = 0
    for link in speech_links:
        i += 1

        r = requests.get(link)
        soup = bs.BeautifulSoup(r.text, features="lxml")

        div = soup.find('div', { 'class': 'pad' })

        if div == None:
            div = soup.findAll('div', { 'class': 'field-item even' })[2]

        confirmed = 'GUD BEVARE DANMARK' in div.text

        if not confirmed:
            div = soup.findAll('div', { 'class': 'field field-name-field-left-collumn field-type-text-long field-label-hidden'})[0]

        confirmed = 'GUD BEVARE DANMARK' in div.text

        whole += div.text + '\n'

        print(f"ÅR {2000 + i}: {len(div.text)} tegn // {confirmed}")

    with open(DATA_PATH, "w") as f:
        f.write(whole)

    return whole

file = ""

if os.path.isfile(DATA_PATH):
    with open(DATA_PATH, "r") as f:
        file = f.read()

        file.replace('æ', '$')
        file.replace('ø', '@')
        file.replace('å', '#')

        file = unidecode.unidecode(file)
        file_len = len(file)
else:
    file = queen_spider()
    file_len = len(file)

chars = sorted(list(set(file)))

print(f"Total length of corpus: {len(file)}")
print(f"Total number of unique chars: {len(chars)}")

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print()

maxlen     = 40
step_size  = 3
sentences  = []
next_chars = []

for i in range(0, len(file) - maxlen, step_size):
    sentences.append(file[i:i + maxlen])
    next_chars.append(file[i + maxlen])

print(f"Sentence sequences: {len(sentences)}\n")

print("Doing preprocessing stuff ...")

def char_tensor(bruh):
    tensor = torch.zeros(len(bruh)).long()

    for c in range(len(bruh)):
        try:
            tensor[c] = string.printable.index(bruh[c])
        except:
            continue

    return tensor

def random_set(chunk_len=200, batch_size=100):
    inp    = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)

    for bi in range(batch_size):
        start = random.randint(0, file_len - chunk_len)
        end   = start + chunk_len + 1

        chunk = file[start:end]

        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    
    inp = Variable(inp)
    target = Variable(target)

    return inp, target

print("Doing model ... ( ͡° ͜ʖ ͡°)\n")

torch.manual_seed(1)

class LSTMQueen(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(LSTMQueen, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers    = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded    = self.encoder(input)
        
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))

        return output, hidden
    
    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))

        return output, hidden
    
    def init_hidden(self, batch_size):
        return (
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        )

n_chars = len(string.printable)

if args.based != "":
    decoder = torch.load(args.based)
    decoder.eval()
else:
    decoder = LSTMQueen(
        n_chars,
        100, # the best number le'go
        n_chars,
        2
    )
    

optimizer = optim.Adam(decoder.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def generate(decoder, prime, predict_len=100, temperature=0.8):
    if not prime:
        prime = random.choice(chars)

    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime).unsqueeze(0))

    predicted = prime

    for p in range(len(prime) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
    
    inp = prime_input[:,-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        output_dist = output.data.view(-1).div(temperature).exp()
        top         = torch.multinomial(output_dist, 1)[0]

        pred_char = string.printable[top]
        predicted += pred_char

        inp = Variable(char_tensor(pred_char).unsqueeze(0))

    return predicted


def train(input, target):
    hidden = decoder.init_hidden(100)

    decoder.zero_grad()
    loss = 0.0

    for c in range(len(input)): # just as well could've been `y` ofc
        output, hidden = decoder(input[:,c], hidden)

        loss += criterion(output.view(100, -1), target[:,c])
    
    loss.backward()
    optimizer.step()

    return loss.data.item() / 200

def save():
    path = f"saves/queenrnn-{str(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}.pt"
    torch.save(decoder, path)
    print(f"Saved as `{path}`")

def since(a):
    s = time.time() - a
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)

start = time.time()
loss_avg = 0.0

EPOCHS = 1000

try:
    print("Training ...\n")

    for epoch in range(EPOCHS):
        loss = train(*random_set())
        loss_avg += loss

        print('\033[92m[%s (epoch %d : %d%%) @ loss %.4f]\033[0m' % (since(start), epoch, epoch / EPOCHS * 100, loss))
        
        generated = generate(decoder, None, 500)

        generated = generated.replace('$', 'æ')
        generated = generated.replace('@', 'ø')
        generated = generated.replace('#', 'å')


        print(f"{generated}\n")

        if epoch % 200 == 0:
            print("\033[92mSaving to be safe ...\033[0m")
            save()
            print()


    print("Saving ...")
    save()
except KeyboardInterrupt:
    print("Saving to be safe ...")
    save()