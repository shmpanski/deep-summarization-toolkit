import errno
import logging
import os
from itertools import takewhile

import numpy as np
import torch
from gensim.models import Word2Vec
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

from .base_dataset import SummarizationDataset
from .utils import SentenceIterator, export_embeddings

logger = logging.getLogger(__name__)


class BPEDataset(SummarizationDataset):
    """Summarization dataset with Byte-Pair encoding.

    Args:
        directory (str): Dataset directory.
        prefix (str): Dataset preprocessing prefix.
        part (str): Dataset part name. :attr:`directory` must contain :attr:`part`.tsv file.
          Use `None` for sampling.
        max_sequence_length (int, optional): Defaults to 150. Maximum sequence length.

    Note:
        Use **kwargs to set up preprocessing arguments.
    """

    def __init__(self, directory: str, prefix: str, part: str, max_sequence_length=150, **kwargs):
        self.data_workdir = os.path.join(directory, prefix)
        self.spm_file = os.path.join(self.data_workdir, "spm.model")

        if part is None:
            self._sample_init(self.spm_file, max_sequence_length)
            return

        self.source_part_file = os.path.join(directory, part + ".tsv")
        self.part_file = os.path.join(self.data_workdir, part + ".npy")

        if not self.exist(directory, prefix, part):
            logger.info("Dataset part {}/{} not founded".format(self.data_workdir, part))
            self.preprocess(directory, prefix, part, **kwargs)

        self.data = np.load(self.part_file)

        if "spm" in kwargs:
            logger.info("Use existing spm model")
            self.spm = kwargs["spm"]
        else:
            logger.info("Load spm model")
            self.spm = SentencePieceProcessor()
            self.spm.load(self.spm_file)

        self.pad_symbol = self.spm.pad_id()
        self.eos_symbol = self.spm.eos_id()

        self._len = self.data.shape[0]

        sequence_lens = [
            len(seq) for example in self.data for seq in example
        ]
        self.max_sequence_length = min(max_sequence_length, max(sequence_lens))

    def _sample_init(self, spm_file_name, max_sequence_length):
        if not os.path.exists(spm_file_name):
            raise RuntimeError("Firstly preprocess dataset")

        self.spm = SentencePieceProcessor()
        self.spm.load(spm_file_name)

        self.pad_symbol = self.spm.pad_id()
        self.eos_symbol = self.spm.eos_id()

        self._len = 0
        self.data = []

        self.max_sequence_length = max_sequence_length

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self._len

    @staticmethod
    def exist(directory: str, prefix: str, part: str) -> bool:
        """Check dataset existence,

        Args:
            directory (str): Dataset directory.
            prefix (str): Dataset prefix.
            part (str): Dataset part.
            spm_filename (str, optional): Defaults to "spm.model". Name of sentencepiece serialized model.

        Returns:
            bool: Existence status.
        """

        data_workdir = os.path.join(directory, prefix)
        part_filename = os.path.join(data_workdir, part + ".npy")
        spm_filename = os.path.join(data_workdir, "spm.model")

        necessary_files = [part_filename, spm_filename]
        existing = [os.path.exists(filename) for filename in necessary_files]
        return all(existing)

    @staticmethod
    def preprocess(directory: str,
                   prefix: str,
                   part: str,
                   spm: SentencePieceProcessor = None,
                   pretrain_emb=True,
                   vocab_size=30000,
                   embedding_size=300,
                   max_sentence_length=16384,
                   workers=3,
                   skip_gramm=False):
        """Preprocess dataset.

        Args:
            directory (str): Dataset directory.
            prefix (str): Dataset preprocessing prefix.
            part (str): Dataset part. :attr:`directory` must contain :attr:`part`.tsv file with data.
            spm (SentencePieceProcessor, optional): Defaults to None. Sentecepiece model.
            pretrain_emb (bool, optional): Defaults to True. Whether to pretrain embeddings.
            vocab_size (int, optional): Defaults to 30000. Vocabulary size.
            embedding_size (int, optional): Defaults to 300. Pretrained embedding size.
            max_sentence_length (int, optional): Defaults to 16384. Maximum sentence length for sentencepiece.
            workers (int, optional): Defaults to 3. Number of workers.
            skip_gramm (bool, optional): Defaults to False. Whether to use skip-gram type of Word2Vec training.

        Raises:
            FileNotFoundError: Raises if source data file doesn't exist.
        """

        data_workdir = os.path.join(directory, prefix)
        part_source_filename = os.path.join(directory, part + ".tsv")
        part_exported_filename = os.path.join(data_workdir, part + ".npy")
        spm_filename = os.path.join(data_workdir, "spm.model")
        spm_directory = os.path.join(data_workdir, "spm")
        w2v_model_filename = os.path.join(data_workdir, "word2vec.model")
        embeddings_filename = os.path.join(data_workdir, "embedding.npy")

        logger.info("Preprocess {}/{} dataset.".format(data_workdir, part))
        os.makedirs(data_workdir, exist_ok=True)

        if not os.path.exists(part_source_filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), part_source_filename)

        if part not in ["train", "dev"]:
            assert spm is not None, "For non train part, `spm` must be specified."
        else:
            logger.info("Start training sentencepiece")
            spm_params = (
                "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
                "--input={} --model_prefix={} --vocab_size={} --max_sentence_length={}".format(
                    part_source_filename,
                    spm_directory,
                    vocab_size,
                    max_sentence_length
                )
            )
            SentencePieceTrainer.Train(spm_params)
            spm = SentencePieceProcessor()
            spm.load(spm_filename)

            if pretrain_emb:
                logger.info("Start training Word2Vec embeddings")

                train_senteces = SentenceIterator(part_source_filename, spm)
                logger.info("Loaded train sentences")
                w2v_model = Word2Vec(train_senteces, min_count=0, workers=workers,
                                     size=embedding_size, sg=int(skip_gramm))
                w2v_model.save(w2v_model_filename)

                # Export embeddings
                logger.info("Export embeddings")
                export_embeddings(embeddings_filename, spm, w2v_model)
                logger.info("Embeddings have been saved into {}".format(embeddings_filename))

        logger.info("Start exporting data file")
        sentence_iterator = SentenceIterator(part_source_filename, spm)
        sentence_iterator.export(part_exported_filename)
        logger.info("{} exported".format(part_exported_filename))

    def get_embeddings(self) -> np.array:
        """Load pretrain embeddings.
        Returns:
            np.array: Array with word2vec embeddings if this one exists, otherwise `None`.
        """

        embedinds_path = os.path.join(self.data_workdir, "embedding.npy")
        if not os.path.exists(embedinds_path):
            logging.info("Embedding file does not founded")
            return None
        else:
            logging.info("Loading embedding dump file")
            return np.load(embedinds_path)

    def get_spm(self) -> SentencePieceProcessor:
        return self.spm

    def encode(self, sequences):
        sequences = [self.spm.EncodeAsIds(s)[:self.max_sequence_length] for s in sequences]
        return torch.LongTensor(sequences)

    def decode(self, sequences):
        sequences = [list(takewhile(lambda x: x != self.eos_symbol, sequence)) for sequence in sequences]
        return [self.spm.DecodeIds([token.item() for token in sentence])
                for sentence in sequences]

    def collate_function(self, batch):
        src_list, src_length_list = self._pad_sequence(
            [example[0][:self.max_sequence_length] for example in batch], self.pad_symbol)
        trg_list, trg_length_list = self._pad_sequence(
            [example[1][:self.max_sequence_length] for example in batch], self.pad_symbol)
        batch = {
            "src": torch.LongTensor(src_list),
            "trg": torch.LongTensor(trg_list),
            "src_length": src_length_list,
            "trg_length": trg_length_list,
        }
        return batch

    @staticmethod
    def _pad_sequence(sequences, pad_symbol=0):
        sequence_lengths = [len(sequence) for sequence in sequences]
        max_len = max(sequence_lengths)
        for i, length in enumerate(sequence_lengths):
            to_add = max_len - length
            sequences[i] += [pad_symbol] * to_add
        return sequences, sequence_lengths
