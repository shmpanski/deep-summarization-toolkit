from typing import List, Tuple

import torch


class BeamSearch:
    """BeamSearch Decoder.

    Args:
        k (int): Beam size.
        device (str, optional): Defaults to 'cpu'. Selected device.
    """

    def __init__(self, k: int, device='cpu'):
        self.k = k
        self.device = device
        self.scores = None
        self.sequences = None

    def initial_update(self, probs: torch.FloatTensor):
        scores = -torch.log(probs)
        top_scores, top_tokens = scores.topk(self.k)
        self.sequences = [[token] for token in top_tokens]
        self.scores = top_scores.view(self.k, 1)

    def update(self, probs: torch.FloatTensor):
        """Update beam.

        Args:
            probs (torch.FloatTensor): Probability distribution of vocabulary for each beam
              of shape ``(k, vocab_size)``. For initial update shape must be ``(vocab_size, )``.
        """
        if self.scores is None:
            assert len(probs.shape) == 1, "Initial update must be done with single-beam prob distribution"
            self.initial_update(probs)
            return
        else:
            assert len(probs.shape) == 2, "Update probs must be a matrix of sizes ``(k, vocab)``"
            assert probs.shape[0] == self.k, "Update must be done with k-beam prob distribution"

        probs_scores = self.scores - torch.log(probs)
        top_k_scores, top_k_tokens = probs_scores.topk(self.k)
        top_k_seq_idx = torch.arange(self.k).view(self.k, 1).repeat(1, self.k)
        top_k_scores, top_k_tokens, top_k_seq_idx = [t.view(-1) for t in [top_k_scores, top_k_tokens, top_k_seq_idx]]
        top_scores, indices = top_k_scores.topk(self.k)
        top_tokens = top_k_tokens.take(indices)
        top_seq_idx = top_k_seq_idx.take(indices)

        _sequences = [[]] * self.k
        for i, seq_idx in enumerate(top_seq_idx):
            _sequences[i] = self.sequences[seq_idx] + [top_tokens[i]]

        self.sequences = _sequences
        self.scores = top_scores.view(self.k, 1)

    def search(self) -> torch.LongTensor:
        """Find best ``k`` sequence.

        Returns:
            torch.LongTensor: Decoded sequences.
        """

        return torch.LongTensor(self.sequences, device=self.device)
