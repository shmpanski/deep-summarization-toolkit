from typing import Tuple

import torch
from ignite.engine import Engine
from torch import nn, optim

from dst.data import BPEDataset
from dst.data.decoding import BeamSearch
from dst.models import BaseSummarizationModel
from dst.nn import (PBATransformerDecoderLayer, PBATransformerEncoderLayer,
                    PositionalEmbedding)
from dst.nn.utils import autoregressive_mask_like


class PBATransformer(BaseSummarizationModel):
    def __init__(self,
                 max_seq_len: int,
                 vocab_size: int,
                 embedding_weights: torch.Tensor = None,
                 num_layers=6,
                 emb_size=600,
                 dim_m=512,
                 dim_i=2048,
                 attention="heterogeneous",
                 initial_token_idx=2,
                 dropout=0.1,
                 **kwargs):
        super(PBATransformer, self).__init__()

        if attention == "interleaved":
            n_heads = 1
        elif 'n_heads' in kwargs:
            n_heads = kwargs['n_heads']
        else:
            n_heads = sum(kwargs['head_convs'])
        assert dim_m % n_heads == 0, "Model dimension must be divisible by total count of attention heads"

        dim_proj = dim_m // n_heads

        self.attention = attention
        self.initial_token_idx = initial_token_idx
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        encoder_decoder_args = {
            'dim_m': dim_m,
            'dim_proj': dim_proj,
            'dim_i': dim_i,
            'dropout': dropout,
            'attention': attention,
            **kwargs
        }

        if embedding_weights is not None:
            vocab_size, emb_size = embedding_weights.shape
        self.vocab_size = vocab_size

        self.embedding = PositionalEmbedding(max_seq_len,
                                             dim_m,
                                             vocab_size,
                                             emb_size,
                                             embedding_weights)
        self.encoder_layers = nn.ModuleList([
            PBATransformerEncoderLayer(**encoder_decoder_args) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            PBATransformerDecoderLayer(**encoder_decoder_args) for _ in range(num_layers)
        ])
        self.out = nn.Linear(dim_m, vocab_size)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        encoder_state = self.embedding(source)
        for encoder_layer in self.encoder_layers:
            encoder_state = encoder_layer(encoder_state)

        decoder_state = self.embedding(target)
        for decoder_layer in self.decoder_layers:
            decoder_state = decoder_layer(decoder_state, encoder_state)

        output = self.out(decoder_state)
        return output

    def inference(self, source: torch.Tensor, limit: int, beam_size=5) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = source.shape[0]

        encoder_state = self.embedding(source)

        for encoder_layer in self.encoder_layers:
            encoder_state = encoder_layer(encoder_state)

        initial_tokens = 2 if self.attention == "interleaved" else 1
        initial_distribution = torch.zeros(self.vocab_size, device=source.device)
        initial_distribution[self.initial_token_idx] = 1.0
        generated_seq = torch.full((batch_size, beam_size, initial_tokens),
                                   self.initial_token_idx,
                                   dtype=torch.long,
                                   device=source.device)

        batch_beams = [BeamSearch(beam_size, source.device) for _ in range(batch_size)]
        for beam in batch_beams:
            beam.update(initial_distribution)

        for _ in range(limit):
            candidate_seqs = generated_seq.transpose(0, 1)
            candidate_distributions_list = []
            for candidate_seq in candidate_seqs:
                decoder_state = self.embedding(candidate_seq)

                for decoder_layer in self.decoder_layers:
                    decoder_state = decoder_layer(decoder_state, encoder_state)

                candidate_out_distr = self.out(decoder_state).softmax(-1)
                candidate_distributions_list.append(candidate_out_distr)
            # (batch, beam, t, vocab_size)
            candidate_distributions = torch.stack(candidate_distributions_list, 1)

            generated_seq_list = []
            for batch_id, beam in enumerate(batch_beams):
                beam.update(candidate_distributions[batch_id, :, -1])
                generated_seq_list.append(beam.search())

            # (batch, beam, t)
            generated_seq = torch.stack(generated_seq_list)

            if source.device.type != 'cpu':
                torch.cuda.empty_cache()

        generated_seq = generated_seq[:, 0, 1:].contiguous()
        output_distr = candidate_distributions[:, 0].contiguous()
        return generated_seq, output_distr

    def create_trainer(self, optimizer: optim.Optimizer, device: torch.device) -> Engine:
        """Create :class:`ignite.Engine` trainer.

        Args:
            optimizer (optim.Optimizer): torch optimizer.
            device (torch.device): selected device.

        Returns:
            Engine: training Engine.
        """

        def _update(engine: Engine, batch: dict):
            batch["src"] = batch["src"].to(device)
            batch["trg"] = batch["trg"].to(device)

            self.train()
            optimizer.zero_grad()
            gen_probs = self.forward(batch["src"], batch["trg"])[:, :-1, :].contiguous()
            loss = self.criterion(gen_probs.view(-1, self.vocab_size),
                                  batch["trg"][:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            return loss.item()
        return Engine(_update)

    def create_evaluator(self, device: torch.device, **kwargs) -> Engine:
        """Create :class:`ignite.Engine` evaluator

        Args:
            device (torch.device): selected device.

        Returns:
            Engine: evaluator engine.
        """

        def _evaluate(engine: Engine, batch: dict):
            batch["src"] = batch["src"].to(device)
            batch["trg"] = batch["trg"].to(device)

            self.eval()
            generated, __ = self.inference(batch["src"], batch["trg"].shape[1] - 1, **kwargs)

            return generated, batch["trg"][:, 1:]
        return Engine(_evaluate)

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
        return PBATransformer(**margs), margs
