import csv
import logging
import os
import typing

import nltk
import numpy as np
import urllib
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class SentenceIterator:
    def __init__(self, data_file_name, spm):
        """
        Load and iterate dataset.
        Args:
            data_file_name (str): dataset file name.
            spm (sentencepice.SentencePieceProcessor): Sentencepiece model.
        """
        self.data_file_name = data_file_name
        self.spm = spm
        self.bos_id, self.eos_id = spm.bos_id(), spm.eos_id()

        with open(self.data_file_name) as file:
            reader = csv.reader(file, delimiter="\t")

            self.data = np.array(
                [[[self.bos_id] + self.spm.EncodeAsIds(p) + [self.eos_id]
                  for p in row] for row in reader])

    def __iter__(self):
        for row in self.data:
            for part in row:
                yield list(map(str, part))

    def export(self, filename):
        """
        Export sentence data into .npy file.
        Args:
            filename (str): exported filename.
        """
        np.save(filename, self.data)


def export_embeddings(filename, sp_model, w2v_model):
    """Export embeddings into numpy matrix.

    Args:
        filename (str): the name of the exported file.
        sp_model (sentencepice.SentencePieceProcessor): Sentencepice model.
        w2v_model (gensim.models.Word2Vec): Word2Vec model.
    """
    dim = w2v_model.vector_size
    vocab_size = len(sp_model)
    table = np.array([
        w2v_model[str(i)] if str(i) in w2v_model.wv else np.zeros([dim])
        for i in range(vocab_size)
    ])
    np.save(filename, table)


def split_into_sentences(text: str) -> typing.List[str]:
    """Split text into sentences.

    Args:
        text (str): Text.

    Returns:
        typing.List[str]: Sentences.
    """

    return nltk.sent_tokenize(text)


def download_url(url: str,
                 directory: str,
                 filename: str = None) -> str:
    directory = os.path.normpath(directory)
    if not filename:
        filename = os.path.basename(url)
    filepath = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)

    if not os.path.isfile(filepath):
        logger.info('Downloading ' + url + ' to ' + filepath)
        with tqdm(unit='B', unit_scale=True) as pbar:
            urllib.request.urlretrieve(
                url, filepath,
                reporthook=get_bar_updater(pbar)
            )

    return filepath


def get_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update
