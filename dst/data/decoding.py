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
        self.scores = torch.zeros(k, 1, device=device)
        self.sequences = [[]] * k

    def update(self, probs: torch.FloatTensor):
        """Update beam.

        Args:
            probs (torch.FloatTensor): Probability distribution of vocabulary for each beam
              of shape ``(k, vocab_size)``.
        """

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
        """Find best sequence.

        Returns:
            torch.LongTensor: Decoded sequence.
        """

        idx = self.scores.view(-1).argmax()
        sequence = self.sequences[idx]
        return torch.LongTensor(sequence, device=self.device)
