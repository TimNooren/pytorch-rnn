
from torch import LongTensor
from torch.utils.data import Dataset


class CharDataset(Dataset):

    def __init__(self, file_name, sample_len):
        super(CharDataset, self).__init__()

        self.sample_len = sample_len

        with open(file_name, 'r') as _file:
            self.data = _file.read()

        self.i_to_char = sorted(list(set(self.data)))
        self.char_to_i = {char: i for i, char in enumerate(self.i_to_char)}

        self.input_size = len(self.i_to_char)

    def __len__(self):
        return len(self.data) - self.sample_len

    def __getitem__(self, ix):
        sample = self.data[ix:ix+self.sample_len-1]
        target = self.data[ix+1:ix+self.sample_len]
        return self.to_tensor(sample), self.to_tensor(target)

    def to_tensor(self, seq):
        return LongTensor([self.char_to_i[char] for char in seq])
