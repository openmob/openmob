import torch
import pandas as pd
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequence_length,
    ):
        self.sequence_length = sequence_length
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        train_df = pd.read_csv('../assets/LSTM_None/user_40.csv',index_col=0)#.iloc[0:20000]
        text = train_df.values.reshape(-1)
        return text

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        if self.index_to_word[self.words_indexes[index:index+self.sequence_length][0]] != 'DAY_END':
            return (
                torch.tensor(self.words_indexes[index:index+self.sequence_length]),
                torch.tensor(self.words_indexes[index+1:index+self.sequence_length+1]),
            )
        else:
            return (
                torch.tensor(self.words_indexes[index:index+self.sequence_length]),
                torch.tensor(self.words_indexes[index:index+self.sequence_length]),
            )