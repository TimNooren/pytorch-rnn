
import argparse

from torch import LongTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable

from .model import CharRNN
from .dataset import CharDataset

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', '-f', default='sherlock.txt')
file_name = parser.parse_args().file_name

n_epochs = 10
print_interval = 1000
sample_len = 100
generate_len = 100

hidden_size = 64
embedding_size = 8

dataset = CharDataset(file_name, sample_len)
dataloader = DataLoader(dataset, shuffle=True)

rnn = CharRNN(dataset.input_size, embedding_size, hidden_size)
optimiser = Adam(rnn.parameters())
criterion = CrossEntropyLoss()

for epoch in range(n_epochs):

    print('Starting epoch: ' + str(epoch))

    total_loss = 0

    for i, (sample, target) in enumerate(dataloader):

        predictions = rnn(Variable(sample))

        loss = criterion(predictions, Variable(target.view(-1)))
        total_loss += loss.data[0]

        rnn.zero_grad()
        loss.backward()
        optimiser.step()

        if i % print_interval == 0:
            print(i, total_loss)
            total_loss = 0

            input = Variable(LongTensor(1).random_(dataset.input_size))
            generated = rnn.generate(input, generate_len)
            print(''.join([dataset.i_to_char[i.data[0]] for i in generated]) + '\n')
