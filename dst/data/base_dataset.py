from torch.utils.data import Dataset


class SummarizationDataset(Dataset):
    """Base class for any summarization dataset.

    Notes:
        Summarization dataset must have `__getitem__`, `__len__`, `encode`,
        `decode`, `get_embeddings`, `get_spm` and `collate_function` functions.
    """

    def __init__(self, directory, prefix, part, max_sequence_length, **kwargs):
        raise NotImplementedError()

    def decode(self, sequence):
        raise NotImplementedError()

    def encode(self, sequence):
        raise NotImplementedError()

    def collate_function(self, batch):
        raise NotImplementedError()

    def get_embeddings(self):
        raise NotImplementedError()

    def get_spm(self):
        raise NotImplementedError()
