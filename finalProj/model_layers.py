"""
Model Layers:

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

# Embedding layer:
# word vectors are embedded using GloVe embeddings
# projected using linear transoform and passed on to
# Highway Encoder to get final embedding
class EmbeddingLayer(nn.Module):

    # args:
    # word_vecs: pretrained word vectors
    # hidden_size: size of hidden activation
    # drop_prob: probability of activations going to zero.

    def __init__(self, word_vecs, hidden_size, drop_prop):
        super(EmbeddingLayer, self).__init__()
        self.drop_prop = drop_prop
        self.embeding = nn.Embedding.from_pretrained(word_vecs)
        self.proj = nn.Linear(word_vecs.size(1), hidden_size, bias = False)
        self.highway = HighwayEncoder(2, hidden_size)

    def forward(self, input):
        embed = self.embeding(input)
        embed = F.dropout(embed, self.drop_prop, self.training)
        embed = self.proj(embed)
        embed = self.highway(embed)

        return embed

# Highway Encoder:
# Encodes input embedding
class HighwayEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transf = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, input):
        for gate, transform in zip(self.gates, self.tranf):
            g = torch.sigmoid(gate(input))
            t = F.relu(transform(input))
            output = g * t + (1 - g) * input
        return output

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_prop = 0.):
        super(LSTMEncoder, self).__init__()
        self.drop_prop = drop_prop
        self.lstm = nn.LSTM()
































