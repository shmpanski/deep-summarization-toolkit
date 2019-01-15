from torch.utils.data import Dataset


class SummarizationDataset(Dataset):
    """Base class for any summarization dataset.

    Notes:
        Dataset must loaded in two ways: from source data files and from dumps. This logic speed up
        multiple runnings with equal dataset.
        - First one implements by `preprocess` method. `preporcess` gets directory of sources
          and parts of dataset (at least train and test parts) and returns dataset instances for each part.
          This method also exports preprocessed data in suitable format, that helps load data faster using constructor.
        - Second one implements by default consturctor of class. Constructor also gets directory
          of dumped dataset and uses it's as sources.
    """

    def __init__(self, directory, prefix, part, max_sequence_length):
        raise NotImplementedError

    def decode(self, sequence):
        raise NotImplementedError

    def encode(self, sequence):
        raise NotImplementedError

    def collate_function(batch):
        raise NotImplementedError

    @staticmethod
    def exist(directory, prefix, part):
        raise NotImplementedError

    @staticmethod
    def preprocess(self, directory, prefix, parts, max_sequence_length):
        raise NotImplementedError
