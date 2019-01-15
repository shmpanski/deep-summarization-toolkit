import torch
from ignite.engine import Engine
from torch import nn

from nn.modules import Transformer

from .base_model import BaseSummarizationModel


class SummarizationTransformer(BaseSummarizationModel):
    def __init__(self,
                 max_seq_len,
                 vocab_size,
                 embedding_weights=None,
                 n_layers=6,
                 emb_size=250,
                 dim_m=512,
                 n_heads=8,
                 dim_i=2048,
                 dropout=0.1):
        """Pure transformer model for summarization task. Actually, it's possible to use this model for MT task.

        Args:
            max_seq_len (int): maximum length of input sequences.
            vocab_size (int): vocabulary size.
            embedding_weights (torch.Tensor, optional): float tensor of shape `(vocab_size, dim_m)`, containing
                embedding weights. Embedding size value would inherited from shape of `embedding_weights` tensor.
            n_layers (int, optional): number transformer layers.
            emb_size (int, optional): embedding size. You do not need to specify a value if you are using
              embedding weights.
            dim_m (int, optional): model dimension (hidden or input size).
            n_heads (int, optional): number of attention heads.
            dim_i (int, optional): inner dimension of position-wise sublayer.
            dropout (float, optional): dropout probability.

        Input:
            - **source_seq** of shape `(batch, source_seq_len)`: a long tensor, containing token indexes of
              source sequence.
            - **target_seq** of shape `(batch, target_seq_len)`: (optional) a long tensor, containing token indexes of
              target sequence.
            - **max_target_seq_len** an int (optional): maximum length of generated sequence. If `target_seq` is None
              `max_target_seq_len` must be defined.

        Output:
            - **generated_seq_probs** of shape `(batch, target_seq_len, vocab_size)`: a float tensor, containing token
              probabilities.
            - **generated_seq** of shape `(batch, target_seq_len)`: a long tensor, containing generated token,
              determined by naive argmax encoding.

        Notes:
            - Model dimension `dim_m` must be divisible by `n_heads` without a remainder. It's necessary for calculating
              projection sizes for multi-head attention.
        """

        super(SummarizationTransformer, self).__init__()

        self.vocab_size = vocab_size
        self.initial_token_idx = 2  # 2 for <bos> tag
        # Get initial probabilities for bos token.
        self.initial_probs = self.get_initial_probs(vocab_size, self.initial_token_idx)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        assert dim_m % n_heads == 0, 'Model `dim_m` must be divisible by `n_heads` without a remainder.'

        dim_proj = dim_m // n_heads

        self.transformer = Transformer(
            max_seq_len, vocab_size, emb_size, embedding_weights, n_layers,
            dim_m, dim_proj, dim_proj, n_heads, dim_i, dropout)

    def forward(self, source, target):
        batch_size = source.shape[0]

        self.transformer.reset_encoder_state()
        output = self.transformer(source, target)
        stacked_probs = self.initial_probs.to(source.device).repeat(batch_size, 1, 1)
        shifted = torch.cat((stacked_probs, output[:, :-1, :]), dim=1)
        return shifted, shifted.argmax(-1)

    def inference(self, source, limit):
        batch_size = source.shape[0]
        # Create initial tokens.
        generated_seq = torch.full((batch_size, 1),
                                   self.initial_token_idx,
                                   dtype=torch.long,
                                   device=source.device)

        # It's very important to do this before every train batch cycle.
        self.transformer.reset_encoder_state()
        for __ in range(1, limit):
            # output = self.transformer(source_inp_seq, generated_inp_seq)
            output = self.transformer(source, generated_seq)

            # Take last token probabilities and find it's index.
            generated_token_idx = output[:, -1, :].argmax(dim=-1).unsqueeze(1)

            # Concatenate generated token with sequence.
            generated_seq = torch.cat((generated_seq, generated_token_idx),
                                      dim=-1)

        stacked_probs = self.initial_probs.to(source.device).repeat(batch_size, 1, 1)
        generated_seq_probs = torch.cat((stacked_probs, output), dim=1)
        return generated_seq_probs, generated_seq

    @staticmethod
    def get_initial_probs(vocab_size, initial_token_idx):
        """Generate initial probability distribution for vocabulary.

        Args:
            vocab_size (int): Size of vocabulary.
            initial_token_idx (int): Initial token index.

        Returns:
            torch.Tensor: float tensor of shape `(1, vocab_size)`.

        """
        probs = torch.zeros(1, vocab_size)
        probs[0, initial_token_idx] = 1
        return probs.float()

    def create_trainer(self, optimizer, device):
        """Create Ignite Engine trainer.

        Args:
            optimizer (torch.optim.Optimizer): torch optimizer
            device (torch.device): training device.

        Returns:
            ignite.engine.Engine: Training Engine.
        """

        def _update(engine, batch):
            # Prepare batches
            batch["src"] = batch["src"].to(device)
            batch["trg"] = batch["trg"].to(device)

            self.train()
            optimizer.zero_grad()
            gen_probs, gen = self.forward(batch["src"], batch["trg"])
            loss = self.criterion(gen_probs.view(-1, self.vocab_size), batch["trg"].view(-1))
            optimizer.step()

            return loss.item()
        return Engine(_update)

    def create_evaluator(self, device):
        def _evaluate(engine, batch):
            # Prepare batches
            batch["src"] = batch["src"].to(device)
            batch["trg"] = batch["trg"].to(device)

            self.eval()
            __, generated = self.inference(batch["src"], batch["trg"].shape[1])

            return generated, batch["trg"]
        return Engine(_evaluate)

    @staticmethod
    def create(dataset, margs: dict):
        """Instantiate model with passed args.

        Args:
            dataset (BPEDataset): used dataset.
            margs (dict): model arguments.

        Returns:
            SummarizationTransformer: instantiated model.
        """

        # Load embeddings if exist
        embedding_dump = dataset.get_embeddings()
        if embedding_dump is not None:
            margs["embedding_weights"] = torch.from_numpy(embedding_dump).float()
        # Choose max sequence length
        margs["max_seq_len"] = dataset.max_sequence_length
        margs["vocab_size"] = len(dataset.spm)
        # Create model
        return SummarizationTransformer(**margs), margs
