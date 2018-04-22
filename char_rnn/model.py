
from torch import zeros
from torch.nn import Module, Embedding, Linear, GRU, ReLU, Softmax
from torch.autograd import Variable
from torch.distributions import Categorical


class CharRNN(Module):

    def __init__(self, input_size, embedding_size, hidden_size):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embed = Embedding(input_size, embedding_size)
        self.gru = GRU(embedding_size, hidden_size)
        self.projection = Linear(hidden_size, input_size)

        self.relu = ReLU()
        self.softmax = Softmax()

    def forward(self, input):

        embedding = self.embed(input)
        output, _ = self.gru(embedding.view(-1, 1, self.embedding_size))
        output = self.projection(output.view(-1, self.hidden_size))

        return output

    def generate(self, input, seq_len):

        samples = [input]
        hidden = Variable(zeros(1, 1, self.hidden_size))

        for i in range(seq_len):
            embedding = self.embed(input)
            _, hidden = self.gru(embedding.view(-1, 1, self.embedding_size), hidden)
            output = self.projection(hidden.view(-1, self.hidden_size))
            output = self.softmax(output)

            input = Categorical(output).sample()
            samples.append(input)

        return samples
