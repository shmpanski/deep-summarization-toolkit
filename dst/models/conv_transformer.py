from typing import Tuple

import torch
from ignite import engine
from torch import nn, optim

from dst.data import BPEDataset
from dst.nn.modules import (PositionalEmbedding, TransformerDecoderLayer,
                            TransformerEncoderLayer)

from .base_model import BaseSummarizationModel


class ConvTransformer(BaseSummarizationModel):
    """Transformer with convolutions applied to ``source`` sequence.

    Args:
        max_seq_len (int): Max sequence length.
        vocab_size (int): Vocabulary size.
        embedding_weights (torch.Tensor, optional): Defaults to None. Embeddings tensor. If is `None`, model will train
          embedding weights.
        num_layers (int, optional): Defaults to 6. Number of layers.
        emb_size (int, optional): Defaults to 600. Embedding size.
        dim_m (int, optional): Defaults to 512. Model dimension.
        dim_i (int, optional): Defaults to 2048. Inner model dimension.
        n_heads (int, optional): Defaults to 8. Number of attention ``heads``.
        initial_token_idx (int, optional): Defaults to 2. Initial token index.
        dropout (float, optional): Defaults to 0.1. Dropout probability.

    Inputs: source, target
        - **source** of shape `(batch, source_seq_len)`: a long tensor, containing token indexes of source sequence.
        - **target** of shape `(batch, target_seq_len)`: a long tensor, containing token indexes of target sequence.

    Output: generated_seq_probs
        - **generated_seq_probs** of shape `(batch, target_seq_len - 1, vocab_size)`: a float tensor, containing token
          probabilities, without first initial token.
    """

    def __init__(self,
                 max_seq_len,
                 vocab_size,
                 embedding_weights=None,
                 num_layers=6,
                 emb_size=600,
                 dim_m=512,
                 dim_i=2048,
                 n_heads=8,
                 initial_token_idx=2,
                 dropout=0.1):
        super(ConvTransformer, self).__init__()

        self.initial_token_idx = initial_token_idx
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        message = 'Model `dim_m` must be divisible by `n_heads` without a remainder.'
        assert dim_m % n_heads == 0, message
        dim_proj = dim_m // n_heads

        encoder_decoder_args = {
            'dim_m': dim_m,
            'dim_q_k': dim_proj,
            'dim_v': dim_proj,
            'n_heads': n_heads,
            'dim_i': dim_i,
            'dropout': dropout
        }

        if embedding_weights is not None:
            vocab_size, emb_size = embedding_weights.shape
        self.vocab_size = vocab_size

        self.embedding = PositionalEmbedding(max_seq_len, dim_m, vocab_size, emb_size, embedding_weights)
        # Embedding convolutions
        # In way of stride = 1, and equal `input_len` and `output_len`
        # padding = dilation * (kernel_size - 1) / 2
        self.convolution = nn.Sequential(
            nn.Conv1d(dim_m, dim_m, kernel_size=5, padding=2),
            nn.SELU(),
            nn.Conv1d(dim_m, dim_m, kernel_size=3, dilation=4, padding=4),
            nn.SELU(),
            nn.Conv1d(dim_m, dim_m, kernel_size=3, dilation=16, padding=16),
            nn.SELU(),
        )
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(**encoder_decoder_args) for _ in range(num_layers)
        ])
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(**encoder_decoder_args) for _ in range(num_layers)
        ])
        # In original paper, authors use shared input and output weights.
        # In this implementation different weghts are used.
        self.out = nn.Sequential(
            nn.Linear(dim_m, vocab_size)
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Encode:
        source_embedded = self.embedding(source)
        source_convolved = self.convolution(source_embedded.transpose(1, -1))
        encoder_state = source_convolved.transpose(1, -1).contiguous()
        for encoder_layer in self.encoder_layers:
            encoder_state = encoder_layer(encoder_state)

        # Decode:
        target_embedded = self.embedding(target)
        decoder_state = target_embedded
        mask = self.autoregressive_mask(target)
        for decoder_layer in self.decoder_layers:
            decoder_state = decoder_layer(decoder_state, encoder_state, mask)

        output = self.out(decoder_state)[:, :-1, :]
        return output.contiguous()

    def inference(self, source: torch.Tensor, limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference step.

        Args:
            source (torch.Tensor): source sequence of shape ``(batch, seq)``
            limit (int): generation limit.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: generated sequence of shape ``(batch, limit)``;
              generated sequence distribution of shape ``(batch, limit, vocab_size)``.
        """

        batch_size = source.shape[0]
        # Encode:
        source_embedded = self.embedding(source)
        source_convolved = self.convolution(source_embedded.transpose(1, -1))
        encoder_state = source_convolved.transpose(1, -1).contiguous()
        for encoder_layer in self.encoder_layers:
            encoder_state = encoder_layer(encoder_state)

        generated_seq = torch.full((batch_size, 1),
                                   self.initial_token_idx,
                                   dtype=torch.long,
                                   device=source.device)

        # Decode:
        for _ in range(0, limit):
            generated_embedded = self.embedding(generated_seq)
            mask = self.autoregressive_mask(generated_seq)
            decoder_state = generated_embedded
            for decoder_layer in self.decoder_layers:
                decoder_state = decoder_layer(decoder_state, encoder_state, mask)

            output_distr = self.out(decoder_state)
            last_generated_token_idx = output_distr[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_seq = torch.cat((generated_seq, last_generated_token_idx), dim=-1)
        return generated_seq[:, 1:].contiguous(), output_distr

    @staticmethod
    def autoregressive_mask(tensor):
        """Generate auto-regressive mask for tensor. It's used to preserving the auto-regressive property.

        Args:
            tensor (torch.Tensor): of shape ``(batch, seq_len)``.

        Returns:
            torch.Tensor: a byte mask tensor of shape ``(batch, seq_len, seq_len)`` containing mask for
            illegal attention connections between decoder sequence tokens.

        """
        batch_size, seq_len = tensor.shape
        x = torch.ones(
            seq_len, seq_len, device=tensor.device).tril(-1).transpose(0, 1)

        return x.repeat(batch_size, 1, 1).byte()

    def create_trainer(self, optimizer: optim.Optimizer, device: torch.device) -> engine.Engine:
        """Create :class:`ignite.engine.Engine` trainer.

        Args:
            optimizer (optim.Optimizer): torch optimizer.
            device (torch.device): selected device.

        Returns:
            engine.Engine: training Engine.
        """

        def _update(engine: engine.Engine, batch: dict):
            batch["src"] = batch["src"].to(device)
            batch["trg"] = batch["trg"].to(device)

            self.train()
            optimizer.zero_grad()
            gen_probs = self.forward(batch["src"], batch["trg"])
            loss = self.criterion(gen_probs.view(-1, self.vocab_size),
                                  batch["trg"][:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            return loss.item()
        return engine.Engine(_update)

    def create_evaluator(self, device: torch.device) -> engine.Engine:
        """Create :class:`ignite.engine.Engine` evaluator

        Args:
            device (torch.device): selected device.

        Returns:
            engine.Engine: evaluator engine.
        """

        def _evaluate(engine: engine.Engine, batch: dict):
            batch["src"] = batch["src"].to(device)
            batch["trg"] = batch["trg"].to(device)

            self.eval()
            generated, __ = self.inference(batch["src"], batch["trg"].shape[1] - 1)

            return generated, batch["trg"][:, 1:]
        return engine.Engine(_evaluate)

    @staticmethod
    def create(dataset: BPEDataset, margs: dict):
        """Instatiate model with passed args.

        Args:
            dataset (BPEDataset): used dataset.
            margs (dict): model arguments.

        Returns:
            ConvTransformer: instatiated model.
        """

        # Load embeddings if exist
        embedding_dump = dataset.get_embeddings()
        if embedding_dump is not None:
            margs["embedding_weights"] = torch.from_numpy(embedding_dump).float()
        # Choose max sequence length
        margs["max_seq_len"] = dataset.max_sequence_length
        margs["vocab_size"] = len(dataset.spm)
        # Create model
        return ConvTransformer(**margs), margs
