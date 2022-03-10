import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceTaggingLSTM(nn.Module):
    def __init__(self, dictionary_size, tagset_size, embedding_dim=300):
        super(SequenceTaggingLSTM, self).__init__()

        self.embedding_layer = nn.Embedding(
            dictionary_size,
            embedding_dim,
        )

        self.lstm_layer = nn.LSTM(
            embedding_dim,
            300,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.hidden_to_tag = nn.Linear(300, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm_layer(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden_to_tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
