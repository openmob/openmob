import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init

# class LSTM_plain(nn.Module):
#     def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
#         super(LSTM_plain, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.has_input = has_input
#         self.has_output = has_output
#
#         if has_input:
#             self.input = nn.Linear(input_size, embedding_size)
#             self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         else:
#             self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         if has_output:
#             self.output = nn.Sequential(
#                 nn.Linear(hidden_size, embedding_size),
#                 nn.ReLU(),
#                 nn.Linear(embedding_size, output_size)
#             )
#
#         self.relu = nn.ReLU()
#         # initialize
#         self.hidden = None # need initialize before forward run
#
#         for name, param in self.rnn.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant(param, 0.25)
#             elif 'weight' in name:
#                 nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
#
#     def init_hidden(self, batch_size):
#         return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda(),
#                 Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda())
#
#     def forward(self, input_raw):
#         if self.has_input:
#             input = self.input(input_raw)
#             input = self.relu(input)
#         else:
#             input = input_raw
#         output_raw, self.hidden = self.rnn(input, self.hidden)
#         if self.has_output:
#             output_raw = self.output(output_raw)
#         # return hidden state at each time step
#         return output_raw


class WordLSTM(nn.Module):
    def __init__(self, n_vocab):
        super(WordLSTM, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
            padding_idx = 0
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))
