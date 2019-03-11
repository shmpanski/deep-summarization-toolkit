import torch
from ignite.engine import Engine
from torch import nn

from dst.data.decoding import BeamSearch
from dst.nn import RNNDecoder, RNNEncoder

from .base_model import BaseSummarizationModel


class SummarizationRNN(BaseSummarizationModel):
    """Summarization RNN model.

        Args:
            vocab_size (int): vocabulary size.
            hidden_size (int, optional): Defaults to 500. Model hidden size.
            embedding_size (int, optional): Defaults to 600. Embedding size.
            embedding_weights (torch.FloatTensor, optional): Defaults to None. Embedding 2-dim weights tensor.
              If `None`, model will train embeddings, otherwise use specified tensor as embedding weights.
              See `Notes` for more details.
            num_layers (int, optional): Defaults to 3. Number of recurrent layers.
            rnn (str, optional): Defaults to "GRU". RNN model name. One of ``GRU`` or ``LSTM``
            attention (str, optional): Defaults to "dot". Name of Luong Attention score function.
              One of ``dot`` or ``general``.
            initial_token_idx (int, optional): Defaults to 2. Index of initial token.
            dropout (float, optional): Defaults to 0.1. Dropout probability.

        Inputs: source, target, source_lengths, target_lengths
            - **source** (torch.LongTensor): Tensor of shape ``(batch, input_seq_len)``.
            - **target** (torch.LongTensor): Tensor of shape ``(batch, target_seq_len)``.
            - **source_lengths** (iterable, optional): list or torch.Tensor, containing lengths of source sequences for
              each batch in decreasing order.
            - **target_lengths** (iterable, optional): list or torch.Tensor, containing lengths of target sequences for
              each batch in decreasing order.

        Outputs: output_vocab_distr, attention_distr
            - **output_vocab_distr** (torch.Tensor): Tensor of shape ``(batch, target_seq_len - 1, vocab_size),
              containing vocabulary distribution for each word **without first initial token** in predicted sequences.
            - **attention_distr** (torch.Tensor): Tensor of shape ``(batch, target_seq_len, source_seq_len),
              containing attention distribution between input and target sequence.

        Notes:
            - If you use :attr:`embedding_weights`, it must be 2-dim :class:`torch.FloatTensor`.
              It's shape is ``(vocab_size, embedding_size)``. Model :attr:`vocab_size` and :attr:`embedding_size`
              would be inherited from tensor's shapes.

        Raises:
            ValueError: Raises if invalid embedding weights passed.
        """

    def __init__(self,
                 vocab_size,
                 hidden_size=500,
                 embedding_size=600,
                 embedding_weights=None,
                 num_layers=3,
                 rnn="GRU",
                 attention="dot",
                 initial_token_idx=2,
                 dropout=0.1):
        super(SummarizationRNN, self).__init__()
        if embedding_weights is not None:
            if not isinstance(embedding_weights, torch.Tensor) or len(embedding_weights.shape) != 2:
                raise ValueError("`embedding_weights` must be a 2-dim float tensor")
            vocab_size, embedding_size = embedding_weights.shape
            self.embeddings = nn.Embedding(*embedding_weights.shape)
            self.embeddings.weight = nn.Parameter(embedding_weights, requires_grad=False)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_size)

        self.vocab_size = vocab_size
        self.initial_token_idx = initial_token_idx
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.encoder = RNNEncoder(embedding_size, hidden_size, num_layers, rnn, dropout, batch_first=True)
        self.decoder = RNNDecoder(embedding_size, hidden_size, num_layers, rnn, attention, dropout, batch_first=True)
        self.out_to_vocab = nn.Sequential(
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, source, target, source_lengths=None, target_lengths=None):
        source_embedded = self.embeddings(source)
        target_embedded = self.embeddings(target)
        encoder_output, h_e = self.encoder(source_embedded, source_lengths)
        decoder_output, _, attention_distr = self.decoder(target_embedded, encoder_output, h_e, target_lengths)
        output_vocab_distr = self.out_to_vocab(decoder_output)
        return output_vocab_distr[:, :-1, :].contiguous(), attention_distr

    def inference(self, source, limit, source_lengths=None, beam_size=5):
        """Inference sequences using model.

        Args:
            source (torch.LongTensor): Tensor of shape ``(batch, seq)`` containing source sequence.
            limit (int): Sequence generaion limit.
            source_lengths (iterable, optional): Defaults to None. Lengths of source sequences for
              each batch in decreasing order.

        Returns:
            (torch.Tensor, torch.Tensor): [0] tensor of shape ``(batch, limit, vocab_size)`` containing vocabulary
              distribution for each word **without first initial token** in predicted sequences. [1] tensor of shape
              ``(batch, limit)`` containing generated sequences **without first initial token**.
        """

        batch_size = source.shape[0]

        source_embedded = self.embeddings(source)
        encoder_state, h_e = self.encoder(source_embedded, source_lengths)

        initial_distribution = torch.zeros(self.vocab_size, device=source.device)
        initial_distribution[self.initial_token_idx] = 1.0
        generated_seq = torch.full((batch_size, beam_size, 1),
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
                decoder_embedding = self.embeddings(candidate_seq)
                decoder_state, _, _ = self.decoder(decoder_embedding, encoder_state, h_e)
                candidate_out_distr = self.out_to_vocab(decoder_state).softmax(-1)
                candidate_distributions_list.append(candidate_out_distr)
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

    def create_trainer(self, optimizer, device):
        """Create :class:`ignite.engine.Engine` trainer.

        Args:
            optimizer (torch.optim.Optimizer): torch optimizer
            device (torch.device): training device.

        Returns:
            ignite.engine.Engine: Training Engine.
        """
        def _update(engine, batch):
            batch["src"] = batch["src"].to(device)
            batch["trg"] = batch["trg"].to(device)

            self.train()
            optimizer.zero_grad()
            gen_probs, attention_distr = self.forward(batch["src"], batch["trg"])
            loss = self.criterion(gen_probs.view(-1, self.vocab_size),
                                  batch["trg"][:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            return loss.item()
        return Engine(_update)

    def create_evaluator(self, device, **kwargs):
        """Create :class:`ignite.engine.Engine` evaluator.

        Args:
            optimizer (torch.optim.Optimizer): torch optimizer
            device (torch.device): training device.

        Returns:
            ignite.engine.Engine: Evaluator Engine.
        """
        def _evaluate(engine, batch):
            batch["src"] = batch["src"].to(device)
            batch["trg"] = batch["trg"].to(device)

            self.eval()
            generated, _ = self.inference(batch["src"], batch["trg"].shape[1] - 1, **kwargs)

            return generated, batch["trg"][:, 1:]
        return Engine(_evaluate)

    @staticmethod
    def create(dataset, margs):
        """Instantiate model with passed args.

        Args:
            dataset (BPEDataset): used dataset.
            margs (dict): model arguments.

        Returns:
            SummarizationRNN: instantiated model.
        """
        # Load embeddings if exist
        embedding_dump = dataset.get_embeddings()
        if embedding_dump is not None:
            margs["embedding_weights"] = torch.from_numpy(embedding_dump).float()
        # Choose max sequence length
        margs["vocab_size"] = len(dataset.spm)
        # Create model
        return SummarizationRNN(**margs), margs
