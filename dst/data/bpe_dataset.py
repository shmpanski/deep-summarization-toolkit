import errno
import logging
import os
from functools import reduce
from operator import and_
from itertools import takewhile

import numpy as np
import sentencepiece
import torch
from gensim.models import Word2Vec

from .base_dataset import SummarizationDataset
from .utils import SentenceIterator, export_embeddings


class BPEDataset(SummarizationDataset):
    def __init__(self, directory, prefix, part, max_sequence_length=150):
        data_file_name = os.path.join(directory, prefix, part + ".npy")
        spm_file_name = os.path.join(directory, prefix, "spm.model")

        self.directory = directory
        self.prefix = prefix
        self.part = part
        self.__data = np.load(data_file_name)
        self.spm = sentencepiece.SentencePieceProcessor()
        self.spm.load(spm_file_name)
        self.pad_symbol = self.spm.pad_id()
        self.eos_symbol = self.spm.eos_id()
        self.__len = self.__data.shape[0]
        self.limit = max_sequence_length

        sequence_lens = [
            len(seq) for example in self.__data for seq in example
        ]
        self.max_sequence_length = min(self.limit, max(sequence_lens))

    def __getitem__(self, index):
        return self.__data[index]

    def __len__(self):
        return self.__len

    def decode(self, sequences):
        sequences = [list(takewhile(lambda x: x != self.eos_symbol, sequence)) for sequence in sequences]
        return [self.spm.DecodeIds([token.item() for token in sentence])
                for sentence in sequences]

    def get_embeddings(self):
        """Load pretrain embeddins.

        Returns:
            np.array: Array with word2vec embeddings if this one exists, otherwise `None`.
        """

        embedinds_path = os.path.join(self.directory, self.prefix, "embedding.npy")
        if not os.path.exists(embedinds_path):
            logging.info("Embedding file does not founded")
            return None
        else:
            logging.info("Loading embedding dump file")
            return np.load(embedinds_path)

    def encode(self, sequences):
        raise NotImplementedError

    @staticmethod
    def exist(directory, prefix, parts):
        """Check dataset dump existence.

        Args:
            directory (str): Dataset directory.
            prefix (str): Dataset preprocess prefix.
            parts (list): Dataset parts.

        Returns:
            bool: True if all dump files existed.
        """
        if isinstance(parts, str):
            parts = [parts]
        parts_file_name = [os.path.join(directory, prefix, part + ".npy") for part in parts]
        smp_file_name = os.path.join(directory, prefix, "spm.model")

        necessary_files = parts_file_name + [smp_file_name]
        existing = [os.path.exists(filename) for filename in necessary_files]
        return reduce(and_, existing)

    @staticmethod
    def __pad_sequence(sequences, pad_symbol=0):
        sequence_lengths = [len(sequence) for sequence in sequences]
        max_len = max(sequence_lengths)
        for i, length in enumerate(sequence_lengths):
            to_add = max_len - length
            sequences[i] += [pad_symbol] * to_add
        return sequences, sequence_lengths

    def collate_function(self, batch):
        src_list, src_length_list = BPEDataset.__pad_sequence(
            [example[0][:self.limit] for example in batch], self.pad_symbol)
        trg_list, trg_length_list = BPEDataset.__pad_sequence(
            [example[1][:self.limit] for example in batch], self.pad_symbol)
        batch = {
            "src": torch.LongTensor(src_list),
            "trg": torch.LongTensor(trg_list),
            "src_length": src_length_list,
            "trg_length": trg_length_list,
        }
        return batch

    @staticmethod
    def preprocess(directory, prefix, parts: list, max_sequence_length=150,
                   pretrain_emb=True, vocab_size=3000, embedding_size=600,
                   max_sentence_length=16384, workers=3, skip_gramm=False):
        """Preprocess dataset.

        Args:
            directory (str): Dataset directory, containing .tsv parts.
            prefix (str): Dataset preprocessing prefix.
            parts (list): Parts of dataset.
            pretrain_emb (bool, optional): Defaults to True. Whether to use pretrained embeddings.
            vocab_size (int, optional): Defaults to 3000. Vocabulary size.
            embedding_size (int, optional): Defaults to 600. Embedding size.
            max_sentence_length (int, optional): Defaults to 16384. Sentecepiece max seq length
            workers (int, optional): Defaults to 3. Number of workers.
            skip_gramm (bool, optional): Defaults to False. Whether to use skip-gram word2vec.

        Raises:
            LookupError: Parts must contain `train`. Otherwise exception raised.
            FileNotFoundError: Raises if can not find dataset files.

        Returns:
            tuple: Tuple of BPEDataset for each dataset part respectively.
        """

        if 'train' not in parts:
            raise LookupError("There is not `train` part of dataset. Can not train sentencepiece and word2vec")

        # Check data files existing
        train_part_file = os.path.join(directory, "train.tsv")
        workdir = os.path.join(directory, prefix)
        data_part_files = [os.path.join(directory, part + ".tsv") for part in parts]
        for part_file in data_part_files:
            if not os.path.exists(part_file):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), part_file)

        # Create folder
        os.makedirs(workdir, exist_ok=True)

        # Train sentecepiece:
        logging.info("Start training sentecepiece")
        spm_directory = os.path.join(workdir, "spm")
        spm_params = (
            "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
            "--input={} --model_prefix={} --vocab_size={} --max_sentence_length={}".format(
                train_part_file, spm_directory, vocab_size, max_sentence_length
            )
        )
        sentencepiece.SentencePieceTrainer.Train(spm_params)
        spm = sentencepiece.SentencePieceProcessor()
        spm.load(spm_directory + ".model")

        if pretrain_emb:
            # Train word2vec
            logging.info("Start training word2vec")
            train_senteces = SentenceIterator(train_part_file, spm)
            logging.info("Loaded train senteces")
            w2v_model = Word2Vec(train_senteces, min_count=0, workers=workers, size=embedding_size, sg=int(skip_gramm))
            w2v_model_filename = os.path.join(workdir, "word2vec.model")
            w2v_model.save(w2v_model_filename)

            # Export embeddings
            logging.info("Export embeddings")
            embeddings_filename = os.path.join(workdir, "embedding.npy")
            export_embeddings(embeddings_filename, spm, w2v_model)
            logging.info("Embeddings have been saved into {}".format(embeddings_filename))

        logging.info("Start exporting data files")
        for part in parts:
            # Export each part of dataset into `part.npy`
            source_file_name = os.path.join(directory, part + ".tsv")
            exported_file_name = os.path.join(workdir, part + ".npy")
            sentence_iterator = SentenceIterator(source_file_name, spm)
            sentence_iterator.export(exported_file_name)
            logging.info("{} exported".format(exported_file_name))
        logging.info("Data preprocessing completed")

        return tuple(BPEDataset(directory, prefix, part, max_sequence_length) for part in parts)
