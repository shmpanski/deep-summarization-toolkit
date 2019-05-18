import torch
from ignite.engine import Engine
from torch import nn, optim

from dst.nn import PositionalEmbedding, TransformerDecoderLayer, TransformerEncoderLayer
from dst.models import BaseSummarizationModel
from dst.nn.modules.loss import NLLvMF
from dst.nn.utils import autoregressive_mask_like
from dst.data import BPEDataset


class VMFTransformer(BaseSummarizationModel):
    """Transformer model with NLLvMF loss.

    Args:
        max_seq_len (int): maximum length of input sequences.
        emb_weights (torch.Tensor, optional): :class:`torch.FloatTensor` of shape ``(vocab_size, dim_m)``,
            containing word embedding weights. Defaults to None.
        vocab_size (int, optional): vocabulary size. Defaults to 30000.
        emb_size (int, optional): embedding size. Defaults to 250.
        n_layers (int, optional): number of transformer layers. Defaults to 6.
        n_heads (int, optional): number of dot-product attention heads. Defaults to 8.
        dim_m (int, optional): model dimension. Defaults to 512.
        dim_i (int, optional): inner dimension of position-wise sublayer. Defaults to 2048.
        initial_token_idx (int, optional): initial token index. Defaults to 2.
        dropout (float, optional): dropout probability. Defaults to 0.1.
        l1 (float, optional): l1 penalty term. Defaults to 0.02.
        l2 (float, optional): l2 penalty term. Defaults to 0.1.

    Input:
        - **input** of shape ``(batch, input_seq_len)``: a :class:`torch.LongTensor`, containing token indexes of
            source sequence.
        - **output** of shape ``(batch, output_seq_len)``: a :class:`torch.LongTensor`, containing token indexes of
            target sequence.

    Output:
        - **output_distribution** of shape ``(batch, output_seq_len-1, emb_size)``: a :class:`torch.FloatTensor`,
            containing predicted output vectors, without first <initial> token.

    Notes:
        - if :attr:`emb_weights` are given, :attr:`vocab_size` and :attr:`emb_size` will inherited from
            :attr:`emb_weights` shape.

    """

    def __init__(
        self,
        max_seq_len,
        emb_weights=None,
        vocab_size=30000,
        emb_size=250,
        n_layers=6,
        n_heads=8,
        dim_m=512,
        dim_i=2048,
        initial_token_idx=2,
        dropout=0.1,
        l1=0.02,
        l2=0.1
    ):
        super(VMFTransformer, self).__init__()

        self.initial_token_idx = initial_token_idx
        self.criterion = NLLvMF()

        message = "Model `dim_m` must be divisible by `n_heads` without a remainder."
        assert dim_m % n_heads == 0, message
        dim_proj = dim_m // n_heads

        if emb_weights is not None:
            assert isinstance(
                emb_weights, torch.Tensor
            ), "Embedding weights must be a `torch.Tensor`"
            vocab_size, emb_size = emb_weights.shape
            emb_weights = nn.functional.normalize(emb_weights)

        enc_dec_args = {
            "dim_m": dim_m,
            "dim_q_k": dim_proj,
            "dim_v": dim_proj,
            "n_heads": n_heads,
            "dim_i": dim_i,
            "dropout": dropout,
        }

        # Transformer Encoder:
        self.emb = PositionalEmbedding(
            max_seq_len, dim_m, vocab_size, emb_size, emb_weights  # , max_norm=1
        )
        self.enc_layers = nn.ModuleList(
            [TransformerEncoderLayer(**enc_dec_args) for _ in range(n_layers)]
        )
        # Transformer Decoder:
        self.dec_layers = nn.ModuleList(
            [TransformerDecoderLayer(**enc_dec_args) for _ in range(n_layers)]
        )
        # Project into embedding space
        self.out = nn.Linear(dim_m, emb_size)
        self.emb_weights = self.emb.embedding.weight
        self.emb_size = emb_size

    def forward(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        enc_state = self.emb(input)
        for enc_layer in self.enc_layers:
            enc_state = enc_layer(enc_state)

        mask = self.autoregressive_mask(output)
        dec_state = self.emb(output)
        for dec_layer in self.dec_layers:
            dec_state = dec_layer(dec_state, enc_state, mask)

        output = self.out(dec_state)
        return output[:, :-1, :].contiguous()

    def inference(self, input: torch.LongTensor, limit=15) -> torch.LongTensor:
        """Inference prediction.

        Args:
            input (torch.LongTensor): tensor with input sequences of shape `(batch, input_seq_len)`.
            limit (int, optional): inference limit. Defaults to 15.

        Returns:
            torch.LongTensor: float tensor of shape `(batch, limit)` with generated sequence.
        """
        batch_size = input.shape[0]

        enc_state = self.emb(input)
        for enc_layer in self.enc_layers:
            enc_state = enc_layer(enc_state)

        generated = torch.full(
            (batch_size, 1),
            self.initial_token_idx,
            dtype=torch.long,
            device=input.device,
        )

        for _ in range(limit):
            mask = self.autoregressive_mask(generated)
            dec_state = self.emb(generated)
            for dec_layer in self.dec_layers:
                dec_state = dec_layer(dec_state, enc_state, mask)
            output = self.out(dec_state)
            last_generated = output[:, -1, :].contiguous()
            dot = self.emb_weights.mm(last_generated.t())
            last_generated_idx = dot.argmax(dim=0).view(batch_size, 1)
            generated = torch.cat((generated, last_generated_idx), dim=-1)

        return generated[:, 1:].contiguous()

    def create_trainer(
        self, optimizer: optim.Optimizer, device: torch.device
    ) -> Engine:
        """Create :class:`ignite.Engine` trainer.

        Args:
            optimizer (optim.Optimizer): torch optimizer.
            device (torch.device): selected device.

        Returns:
            Engine: training Engine.
        """

        def _update(engine: Engine, batch: dict):
            src = batch["src"].to(device)
            trg = batch["trg"].to(device)
            embedded_trg = self.emb.embedding(trg)

            self.train()
            optimizer.zero_grad()
            gen_probs = self.forward(src, trg)
            loss = self.criterion(
                gen_probs.view(-1, self.emb_size),
                embedded_trg[:, 1:, :].contiguous().view(-1, self.emb_size),
            )
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
            src = batch["src"].to(device)
            trg = batch["trg"].to(device)

            self.eval()
            generated = self.inference(src, trg.shape[1] - 1, **kwargs)

            return generated, trg[:, 1:]

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
            margs["emb_weights"] = torch.from_numpy(embedding_dump).float()
        # Choose max sequence length
        margs["max_seq_len"] = dataset.max_sequence_length
        margs["vocab_size"] = len(dataset.spm)
        # Create model
        return VMFTransformer(**margs), margs

    @staticmethod
    def autoregressive_mask(tensor):
        """Generate auto - regressive mask for tensor. It's used to preserving the auto - regressive property.
        Args:
            tensor(torch.Tensor): of shape ``(batch, seq_len)``.
        Returns:
            torch.Tensor: a byte mask tensor of shape ``(batch, seq_len, seq_len)`` containing mask for
            illegal attention connections between decoder sequence tokens.
        """
        batch_size, seq_len = tensor.shape
        x = torch.ones(seq_len, seq_len, device=tensor.device).tril(-1).transpose(0, 1)

        return x.repeat(batch_size, 1, 1).byte()
