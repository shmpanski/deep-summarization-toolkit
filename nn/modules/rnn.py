import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nn.modules import LuongAttention

_avaialable_rnns = ["LSTM", "GRU"]


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, rnn="GRU", dropout=0.1, batch_first=False):
        """Attentional seq-to-seq RNN encoder.

        Args:
            input_size (int): Input size.
            hidden_size (int): Hidden size.
            num_layers (int, optional): Defaults to 3. Number of RNN layers.
            rnn (str, optional): Defaults to "GRU". RNN model name. One of ``GRU`` or ``LSTM``.
            dropout (float, optional): Defaults to 0.1. Dropout probability.
            batch_first (bool, optional): Defaults to False. If `True`, use first dimension for batch.

        Inputs: input, input_lengths, h_0
            - **input** (torch.Tensor): Tensor of shape ``(seq, batch, input_size)``.
            - **input_lengths** (iterable, optional): Collection of sequences length for each batch element in
              decreasing order. If not `None`, uses for PackedSequence.
            - **h_0** (torch.Tensor, optional): Tensor of shape ``(num_layers * 2, batch, hidden_size)`` containing
              initial hidden state.

        Outputs: encoder_output, h_n
            - **encoder_output** (torch.Tensor): Tensor of shape ``(seq, batch, hidden_size * 2)`` containing encoder
              sequence representations.
            - **h_n** (torch.Tensor): Tensor of shape ``(num_layers * 2, batch, hidden_size)`` containing the hidden
              state for `t = seq_len`

        Raises:
            ValueError: Raises if :attr:`rnn` is invalid.
        """

        super(RNNEncoder, self).__init__()

        if rnn not in _avaialable_rnns:
            raise ValueError("Invalid RNN type `{}`. Use one of {}.".format(rnn, _avaialable_rnns))

        self.rnn = nn.RNNBase(rnn, input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True,
                              batch_first=batch_first)

    def forward(self, input, input_lengths=None, h_0=None):
        packed = input_lengths is not None
        if packed:
            input = pack_padded_sequence(input, input_lengths)
        encoder_output, h_n = self.rnn(input, h_0)
        if packed:
            encoder_output, _ = pad_packed_sequence(encoder_output)
        return encoder_output, h_n


class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, rnn="GRU", attention="dot", dropout=0.1,
                 batch_first=False):
        """Attentional seq-to-seq RNN decoder.

        Args:
            input_size (int): Input size.
            hidden_size (int): Hidden size.
            num_layers (int, optional): Defaults to 3. Number of layers.
            rnn (str, optional): Defaults to "GRU". RNN model name. One of ``GRU`` or ``LSTM``.
            attention (str, optional): Defaults to "dot". Attention score function. One of ``dot`` or ``general``.
              See :class:`nn.LuongAttention` for more details.
            dropout (float, optional): Defaults to 0.1. Dropout probability.
            batch_first (bool, optional): Defaults to False. If `True`, use first dimension for batch.

        Inputs: input, encoder_output, h_0, input_lengths
            - **input** (torch.Tensor): Tensor of shape ``(input_seq, batch, input_size)``.
            - **encoder_output** (torch.Tensor): Tensor of shape ``(encoder_seq, batch, hidden_size * 2)``
              containing encoder sequence representations.
            - **input_lengths** (iterable, optional): Collection of sequences length for each batch element in
              decreasing order. If not `None`, uses for PackedSequence.
            - **h_0** (torch.Tensor, optional): Tensor of shape ``(num_layers * 2, batch, hidden_size)``
              containing initial hidden state.

        Outputs: output, attention_distr
            - **output** (torch.Tensor): Tensor of shape ``(input_seq, batch, hidden_size * 2)``.
            - **attention_distr** (torch.Tensor): Tensor of shape ``(batch, input_seq, encoder_seq)``
              containing attention ditribution between :attr:`query` and :attr:`value`.

        Raises:
            ValueError: Raises if :attr:`rnn` or :attr:`attention` are invalid.
        """

        super(RNNDecoder, self).__init__()

        if rnn not in _avaialable_rnns:
            raise ValueError("Invalid RNN type `{}`. Use one of {}.".format(rnn, _avaialable_rnns))

        self.batch_first = batch_first
        self.rnn = nn.RNNBase(rnn, input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True,
                              batch_first=batch_first)
        self.attention = LuongAttention(attention, batch_first=batch_first,
                                        query_size=hidden_size * 2,
                                        key_size=hidden_size * 2)
        self.out = nn.Sequential(
            nn.Linear(4 * hidden_size, 2 * hidden_size),
            nn.SELU()
        )

    def forward(self, input, encoder_output, input_lengths=None, h_0=None):
        packed = input_lengths is not None
        if packed:
            input = pack_padded_sequence(input, input_lengths)
        decoder_output, _ = self.rnn(input, h_0)
        if packed:
            decoder_output, _ = pad_packed_sequence(decoder_output)
        attention, attention_distr = self.attention(encoder_output, encoder_output, decoder_output)
        if not self.batch_first:
            decoder_output = decoder_output.transpose(0, 1)
            output = self.out(torch.cat((attention, decoder_output), -1)).transpose(0, 1)
        else:
            output = self.out(torch.cat((attention, decoder_output), -1))

        return output, attention_distr
