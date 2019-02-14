from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax, pad
from torch.nn.init import kaiming_normal_

from dst.nn.utils import autoregressive_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_q_k):
        """Scaled Dot-Product Attention model: :math:`softmax(QK^T/sqrt(dim))V`.

        Args:
            dim_q_k (int): dimension of `queries` and `keys`.

        Inputs: query, key, value, mask
            - **value** of shape `(batch, seq_len, dim_v)`:  a float tensor containing `value`.
            - **key** of shape `(batch, seq_len, dim_q_k)`: a float tensor containing `key`.
            - **query** of shape `(batch, q_len, dim_q_k)`: a float tensor containing `query`.
            - **mask** of shape `(batch, q_len, seq_len)`, default None: a byte tensor containing mask for
              illegal connections between query and value.

        Outputs: attention, attention_weights
            - **attention** of shape `(batch, q_len, dim_v)` a float tensor containing attention
              along `query` and `value` with the corresponding `key`.
            - **attention_weights** of shape `(batch, q_len, seq_len)`: a float tensor containing distribution of
              attention weights.
        """
        super(ScaledDotProductAttention, self).__init__()

        self.scale_factor = np.power(dim_q_k, -0.5)

    def forward(self, value, key, query, mask=None):
        # (batch, q_len, seq_len)
        adjacency = query.bmm(key.transpose(1, 2)) * self.scale_factor

        if mask is not None:
            adjacency.data.masked_fill_(mask.data, -float('inf'))

        attention = softmax(adjacency, 2)
        return attention.bmm(value), attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim_m, dim_q_k, dim_v, dropout=0.1):
        """Multi-Head Attention model.

        Args:
            n_heads (int): number of heads.
            dim_m (int): hidden size of model.
            dim_q_k (int): dimension of projection `queries` and `keys`.
            dim_v (int): dimension of projection `values`.
            dropout (float, optional): dropout probability.

        Inputs:
            - **value** of shape `(batch, seq_len, dim_m)`: a float tensor containing `value`.
            - **key** of shape `(batch, seq_len, dim_m)`: a float tensor containing `key`.
            - **query** of shape `(batch, q_len, dim_m)`: a float tensor containing `query`.
            - **mask** of shape `(batch, q_len, seq_len)`: default None: a byte tensor containing mask for
              illegal connections between query and value.

        Outputs:
            - **attention** of shape `(batch, q_len, dim_m)`: a float tensor containing attention
              along `query` and `value` with the corresponding `key` using Multi-Head Attention mechanism.
        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.dim_m = dim_m
        self.dim_q_k = dim_q_k
        self.dim_v = dim_v

        self.query_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim_m, dim_q_k))
        self.key_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim_m, dim_q_k))
        self.value_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim_m, dim_v))
        self.attention = ScaledDotProductAttention(dim_q_k)
        self.output = nn.Linear(dim_v * n_heads, dim_m)
        self.dropout = nn.Dropout(dropout)
        self.layer_normalization = nn.LayerNorm(dim_m, eps=1e-12)

        # Initialize projection tensors
        for parameter in [
                self.query_projection, self.key_projection,
                self.value_projection
        ]:
            kaiming_normal_(parameter.data)

    def forward(self, value, key, query, mask=None):
        seq_len = key.shape[1]
        q_len = query.shape[1]
        batch_size = query.shape[0]

        residual = query
        # (batch, x, dim_m) -> (n_heads, batch * x, dim_m)
        value, key, query = map(self.stack_heads, [value, key, query])

        if mask is not None:
            mask = self.stack_mask(mask)

        # (n_heads, batch * x, dim_m) -> (n_heads, batch * x, projection) -> (n_heads * batch, x, projection)
        # where `projection` is `dim_q_k`, `dim_v` for each input respectively.
        value = value.bmm(self.value_projection).view(-1, seq_len, self.dim_v)
        key = key.bmm(self.key_projection).view(-1, seq_len, self.dim_q_k)
        query = query.bmm(self.query_projection).view(-1, q_len, self.dim_q_k)

        # (n_heads * batch, q_len, dim_v)
        context, _ = self.attention(value, key, query, mask)

        # # (n_heads * batch, q_len, dim_v) -> (batch * q_len, n_heads, dim_v) -> (batch, q_len, n_heads * dim_v)
        # context = context.view(self.n_heads, -1, self.dim_v).transpose(0, 1).view(-1, q_len, self.n_heads * self.dim_v)

        # (n_heads * batch, q_len, dim_v) -> (batch, q_len, n_heads * dim_v)
        context_heads = context.split(batch_size, dim=0)
        concat_heads = torch.cat(context_heads, dim=-1)

        # (batch, q_len, n_heads * dim_v) -> (batch, q_len, dim_m)
        out = self.output(concat_heads)
        out = self.dropout(out)

        return self.layer_normalization(out + residual)

    def stack_mask(self, mask):
        """Prepare mask tensor for multi-head Scaled Dot-Product Attention.

        Args:
            mask: A byte tensor of shape `(batch, q_len, seq_len)`.

        Returns:
            A byte tensor of shape `(n_heads * batch, q_len, seq_len)`.
        """
        return mask.repeat(self.n_heads, 1, 1)

    def stack_heads(self, tensor):
        """Prepare tensor for multi-head projection.

        Args:
            tensor: A float input tensor of shape `(batch, x, dim_m)`.

        Returns:
            Stacked input tensor n_head times of shape `(n_heads, batch * x, dim_m)`.
        """
        return tensor.view(-1, self.dim_m).repeat(self.n_heads, 1, 1)


class MultiHeadPhrasalAttentionBase(nn.Module):
    def __init__(self):
        super(MultiHeadPhrasalAttentionBase, self).__init__()

    @staticmethod
    def stack_mask(mask: torch.ByteTensor, n_heads: int) -> torch.ByteTensor:
        """Stack mask for required number of heads.

        Args:
            mask (torch.ByteTensor): Mask tensor of shape ``(batch, x, y)``.
            n_heads (int): Number of heads.

        Returns:
            torch.ByteTensor: Mask tensor of shape ``(n_heads * batch, x, y)``.
        """

        return mask.repeat(n_heads, 1, 1)

    @staticmethod
    def stack_heads(tensor: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Stack inputs for attention heads.

        Args:
            tensor (torch.Tensor): Input tensor of shape ``(batch, seq_len, dim)``.
            n_heads (int): Number of heads.

        Returns:
            torch.Tensor: stacked inputs of shape ``(n_heads, batch, seq_len, dim)``.
        """

        return tensor.repeat(n_heads, 1, 1, 1)

    @staticmethod
    def stack_convolutions(head_convs: Tuple[int, ...], input_dim: int, output_dim: int) -> Iterable[nn.Conv1d]:
        """Generate grouped convolution layers for desired heads.

        Args:
            head_convs (Tuple[int, ...]): Tuple of convolution head distribution.
              For example, (1, 0, 3) means one head of 1 kernel convolution and three heads of 3 kernel convolution.
            input_dim (int): Input dimension.
            output_dim (int): Output dimension

        Returns:
            Iterable[nn.Conv1d]: convoluion modules
        """

        for kernel_size, n_heads in enumerate(head_convs, 1):
            if n_heads != 0:
                yield nn.Conv1d(in_channels=n_heads * input_dim,
                                out_channels=n_heads * output_dim,
                                kernel_size=kernel_size,
                                groups=n_heads)


class MultiHeadHomogeneousAttention(MultiHeadPhrasalAttentionBase):
    """Multi Head Homogeneous Attention.
    See https://arxiv.org/abs/1810.03444 for more details.

    Args:
        dim_m (int): Model dimension.
        dim_proj (int): Projection dimension.
        head_convs (Tuple[int]): Description of convolution distribution per head. For example: (3, 2, 3) means
            three heads with one-kernel convolution, two heads with two-kernel convolution and three heads with
            three-kernel convolutions.
        masked (bool): Defaults to False. Whether to mask illegal connection to sim autoregressive property.
        dropout (float, optional): Defaults to 0.1. Dropout probability.

    Inputs:
        - **value** of shape `(batch, seq_len, dim_m)`: a float tensor containing `value`.
        - **key** of shape `(batch, seq_len, dim_m)`: a float tensor containing `key`.
        - **query** of shape `(batch, q_len, dim_m)`: a float tensor containing `query`.

    Outputs:
        - **attention** of shape `(batch, q_len, dim_m)`: a float tensor containing attention
            along `query` and `value` with the corresponding `key` attention mechanism.
    """

    def __init__(self, dim_m: int, dim_proj: int, head_convs: Tuple[int], masked=False, dropout=0.1):
        super(MultiHeadHomogeneousAttention, self).__init__()

        self.head_convs = head_convs
        self.total_n_heads = sum(head_convs)
        self.dim_m = dim_m
        self.dim_proj = dim_proj
        self.masked = masked

        self.query_projections = nn.ModuleList(self.stack_convolutions((self.total_n_heads, ), dim_m, dim_proj))
        self.value_projections = nn.ModuleList(self.stack_convolutions(head_convs, dim_m, dim_proj))
        self.key_projection = nn.ModuleList(self.stack_convolutions(head_convs, dim_m, dim_proj))
        self.attention = ScaledDotProductAttention(dim_proj)
        self.output = nn.Linear(dim_proj * self.total_n_heads, dim_m)
        self.dropout = nn.Dropout(dropout)
        self.layer_normalization = nn.LayerNorm(dim_m, eps=1e-12)

    def forward(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor) \
            -> torch.Tensor:
        batch_size, q_len, _ = query.shape
        residual = query

        # (batch, q_len, dim_m) -> (n_heads * batch, q_len, dim_proj)
        query = self.calculate_conv_heads(query, self.query_projections, (self.total_n_heads, ))
        query = query.view(self.total_n_heads * batch_size, -1, self.dim_proj)
        # (batch, batch, v_len, dim_m) -> (n_heads * batch, v_len, dim_proj)
        value = self.calculate_conv_heads(value, self.value_projections, self.head_convs)
        value = value.view(self.total_n_heads * batch_size, -1, self.dim_proj)
        # (batch, batch, k_len, dim_m) -> (n_heads * batch, k_len, dim_proj)
        key = self.calculate_conv_heads(key, self.key_projection, self.head_convs)
        key = key.view(self.total_n_heads * batch_size, -1, self.dim_proj)

        if self.masked:
            mask = autoregressive_mask(batch_size, (q_len, q_len), query.device)
            mask = self.stack_mask(mask, self.total_n_heads)
        else:
            mask = None

        # (n_heads * batch, q_len, dim_v)
        context, _ = self.attention(value, key, query, mask)

        # (n_heads * batch, q_len, dim_v) -> (batch, q_len, n_heads * dim_v)
        context_heads = context.split(batch_size, dim=0)
        concat_heads = torch.cat(context_heads, dim=-1)

        # (batch, q_len, n_heads * dim_v) -> (batch, q_len, dim_m)
        out = self.output(concat_heads)
        out = self.dropout(out)

        return self.layer_normalization(out + residual)

    def calculate_conv_heads(self, input: torch.Tensor, conv_layers: Iterable[nn.Conv1d], head_convs: Tuple[int]) \
            -> torch.Tensor:
        """Calculate convolution for heads.

        Args:
            input (torch.Tensor): Input tensor.
            conv_layers (Iterable[nn.Conv1d]): Convolution layers.
            head_convs (Tuple[int]): Tuple of convolution kernels per heads.

        Returns:
            torch.Tensor: Tensor of shape ``(n_heads, batch, seq_len, dim_proj)``.
        """

        conv_layers = iter(conv_layers)
        batch_size, seq_len, dim = input.shape
        # (n_heads, batch, x, dim) -> (batch, n_heads * dim, x)
        heads = []
        for kernel_size, n_heads in enumerate(head_convs, 1):
            if n_heads == 0:
                continue
            # (n_heads, batch, seq_len, dim)
            stacked_input = MultiHeadPhrasalAttentionBase.stack_heads(input, n_heads)
            # (batch, n_heads*dim, seq_len)
            stacked_input = stacked_input.permute(1, 0, 3, 2).contiguous().view(batch_size, n_heads * dim, -1)
            stacked_input = pad(stacked_input, (kernel_size - 1, 0))
            layer = next(conv_layers)
            # (batch, n_heads, dim_proj, convolved_seq_len)
            head = layer(stacked_input).view(batch_size, n_heads, self.dim_proj, -1)
            # (n_heads, batch, convolved_seq_len, dim_proj)
            heads.append(head.permute(1, 0, 3, 2))
        heads = torch.cat(heads, dim=0).contiguous()
        return heads


class MultiHeadHeterogeneousAttention(MultiHeadPhrasalAttentionBase):
    """Multi Head Heterogeneous Attention.
    See https://arxiv.org/abs/1810.03444 for more details.

    Args:
        dim_m (int): Model dimension.
        dim_proj (int): Projection dimension.
        head_convs (Tuple[int]): Description of using convolution stack. For example: (1, 0, 3) means, that in each
          head concatentaion of one and two kernel convolutions representations will be used.
        n_heads (int): Total number of attention heads.
        masked (bool): Defaults to False. Whether to mask illegal connection to sim autoregressive property.
        dropout (float, optional): Defaults to 0.1. Dropout probability.

    Inputs:
        - **value** of shape `(batch, seq_len, dim_m)`: a float tensor containing `value`.
        - **key** of shape `(batch, seq_len, dim_m)`: a float tensor containing `key`.
        - **query** of shape `(batch, q_len, dim_m)`: a float tensor containing `query`.

    Outputs:
        - **attention** of shape `(batch, q_len, dim_m)`: a float tensor containing attention
            along `query` and `value` with the corresponding `key` attention mechanism.
    """

    def __init__(self, dim_m: int, dim_proj: int, head_convs: Tuple[int], n_heads: int, masked=False, dropout=0.1):
        super(MultiHeadHeterogeneousAttention, self).__init__()
        self.total_n_heads = n_heads
        self.dim_m = dim_m
        self.dim_proj = dim_proj
        self.head_convs = tuple((0 if convs == 0 else n_heads for convs in head_convs))
        self.masked = masked

        self.query_projections = nn.ModuleList(self.stack_convolutions((self.total_n_heads, ), dim_m, dim_proj))
        self.value_projections = nn.ModuleList(self.stack_convolutions(self.head_convs, dim_m, dim_proj))
        self.key_projection = nn.ModuleList(self.stack_convolutions(self.head_convs, dim_m, dim_proj))
        self.attention = ScaledDotProductAttention(dim_proj)
        self.output = nn.Linear(dim_proj * self.total_n_heads, dim_m)
        self.dropout = nn.Dropout(dropout)
        self.layer_normalization = nn.LayerNorm(dim_m, eps=1e-12)

    def forward(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor) \
            -> torch.Tensor:
        batch_size, q_len, _ = query.shape
        residual = query

        # (batch, q_len, dim_m) -> (n_heads * batch, q_len, dim_proj)
        query = self.calculate_conv_heads(query, self.query_projections, (self.total_n_heads, ))
        query = query.view(self.total_n_heads * batch_size, -1, self.dim_proj)
        # (batch, batch, v_len, dim_m) -> (n_heads * batch, v_len, dim_proj)
        value = self.calculate_conv_heads(value, self.value_projections, self.head_convs)
        value = value.view(self.total_n_heads * batch_size, -1, self.dim_proj)
        # (batch, batch, k_len, dim_m) -> (n_heads * batch, k_len, dim_proj)
        key = self.calculate_conv_heads(key, self.key_projection, self.head_convs)
        key = key.view(self.total_n_heads * batch_size, -1, self.dim_proj)

        if self.masked:
            mask = autoregressive_mask(batch_size, (q_len, q_len), query.device)
            mask = MultiHeadHeterogeneousAttention.stack_mask(mask, self.head_convs)
        else:
            mask = None

        # (n_heads * batch, q_len, dim_v)
        context, _ = self.attention(value, key, query, mask)

        # (n_heads * batch, q_len, dim_v) -> (batch, q_len, n_heads * dim_v)
        context_heads = context.split(batch_size, dim=0)
        concat_heads = torch.cat(context_heads, dim=-1)

        # (batch, q_len, n_heads * dim_v) -> (batch, q_len, dim_m)
        out = self.output(concat_heads)
        out = self.dropout(out)

        return self.layer_normalization(out + residual)

    def calculate_conv_heads(self, input: torch.Tensor, conv_layers: Iterable[nn.Conv1d], head_convs: Tuple[int]) \
            -> torch.Tensor:
        """Calculate convolution for heads.

        Args:
            input (torch.Tensor): Input tensor.
            conv_layers (Iterable[nn.Conv1d]): Convolution layers.
            head_convs (Tuple[int]): Tuple of convolution kernels per heads.

        Returns:
            torch.Tensor: Tensor of shape ``(n_heads, batch, convolved_multiple_ngrams, dim_proj)``.
        """

        conv_layers = iter(conv_layers)
        batch_size, seq_len, dim = input.shape

        # (n_heads, batch, seq_len, dim)
        stacked_input = MultiHeadPhrasalAttentionBase.stack_heads(input, self.total_n_heads)
        # (batch, n_heads*dim, seq_len)
        stacked_input = stacked_input.permute(1, 0, 3, 2).contiguous().view(batch_size, self.total_n_heads * dim, -1)

        heads = []
        for kernel_size, n_heads in enumerate(head_convs, 1):
            if n_heads == 0:
                continue
            if seq_len < kernel_size:
                break
            layer = next(conv_layers)
            # (batch, n_heads, dim_proj, convolved_seq_len)
            head = layer(stacked_input).view(batch_size, n_heads, self.dim_proj, -1).permute(1, 0, 3, 2)
            # (n_heads, batch, prev_convolved_n-grams + convolved_seq_len, dim_proj)
            heads.append(head)
        return torch.cat(heads, dim=2).contiguous()

    @staticmethod
    def stack_mask(mask: torch.Tensor, head_convs: Tuple[int]) -> torch.Tensor:
        batch_size, q_len, v_len = mask.shape
        stacked_mask = []
        for kernel_size, n_heads in enumerate(head_convs, 1):
            if n_heads == 0:
                continue
            if q_len < kernel_size:
                break
            stacked_mask.append(torch.narrow(mask, -1, kernel_size - 1, v_len - kernel_size + 1))
        stacked_mask = torch.cat(stacked_mask, dim=-1)
        return stacked_mask.repeat(head_convs[0], 1, 1)


class LuongAttention(nn.Module):
    def __init__(self, score="dot", batch_first=True, **kwargs):
        """Luong Attention Model

        Args:
            score (str, optional): Defaults to "dot". One of :attr:`score` function. Available: ``dot`` and ``general``.
            batch_first (bool, optional): Defaults to `True`. If False, swap axis and use ``0`` dimension as sequence.
            key_size (int, optional): Stores in **kwargs. Size of :attr:`key`.
            query_size (int, optional): Stores in **kwargs. Size of :attr:`query`.

        Inputs: value, key, query
            - **value** (torch.Tensor): Tensor of shape ``(batch, seq, value_size)``: Weighted sequence.
            - **key** (torch.Tensor): Tensor of shape ``(batch, seq, key_size)``: Weighing sequence.
            - **query** (torch.Tensor): Tensor of shape ``(batch, q_seq, query_size)``: Query sequence.

        Outputs: attention, attention_distr
            - **attention** (torch.Tensor): Tensor of shape ``(batch, q_seq, value_size)``.
            - **attention_distr** (torch.Tensor): Tensor of shape ``(batch, q_seq, seq)`` containing attention
              ditribution between :attr:`query` and :attr:`value`.

        Note:
            - There is ``concat`` :attr:`score` function, but original formulas are quite strange.
            - In case of use ``dot`` :attr:`score`, :attr:`key_size` and :attr:`query_size` must be equal.

        TODO:
            Need to deal with ``concat`` :attr:`score` function.

        Raises:
            ValueError: Raises if pased invalid :attr:`score` function.
        """

        super(LuongAttention, self).__init__()
        self._available_scores = ["dot", "general"]
        self.batch_first = batch_first
        if score not in self._available_scores:
            raise ValueError("Invalid attention score `{}`. Use one of {}.".format(score, self._available_scores))
        if score == "dot":
            self.score_function = self.score_dot
        if score == "general":
            key_size, query_size = kwargs["key_size"], kwargs["query_size"]
            self.W = nn.Linear(query_size, key_size, bias=False)
            self.score_function = self.score_general

    def score_dot(self, key, query):
        return query.bmm(key.transpose(1, 2))

    def score_general(self, key, query):
        return query.bmm(self.W(key).transpose(1, 2))

    def forward(self, value, key, query):
        if not self.batch_first:
            value = value.transpose(0, 1)
            key = key.transpose(0, 1)
            query = query.transpose(0, 1)
        scores = self.score_function(key, query)
        attention = softmax(scores, 2)
        return attention.bmm(value), attention
