import torch
from torch import nn

from modules import VarDropout


class LM_LSTM_TWO(nn.Module):
    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        tie=True,
        var_drop=True,
        pad_index=0,
        emb_dropout=0.1,
        out_dropout=0.1,
        arch="LSTM",
        n_layers=1,
    ):
        super(LM_LSTM_TWO, self).__init__()

        if tie:
            if hidden_size != emb_size:
                print("WARNING: hidden size and emb size do not match")
            hidden_size = emb_size

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        if arch == "RNN":
            print("Using RNN")
            self.rnn = nn.RNN(
                emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True
            )
        else:
            print("Using LSTM")
            self.rnn = nn.LSTM(
                emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True
            )

        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space
        self.decoder = nn.Linear(hidden_size, output_size)

        if tie:
            self.decoder.weight = self.embedding.weight

        if var_drop:
            print("Using VarDrop")
            self.drop_emb = VarDropout(p=emb_dropout)
            self.drop_dec = VarDropout(p=out_dropout)
        else:
            print("Using Drop")
            self.drop_emb = nn.Dropout(p=emb_dropout)
            self.drop_dec = nn.Dropout(p=out_dropout)

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence)

        dropped_1 = self.drop_emb(embedded)

        rnn_out, _ = self.rnn(dropped_1)

        dropped_2 = self.drop_dec(rnn_out)

        output = self.decoder(dropped_2).permute(0, 2, 1)
        return output
