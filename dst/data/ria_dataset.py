import gzip
import itertools
import json
import logging
import os
import shutil
import tempfile
import typing
import unicodedata
from collections import namedtuple
from csv import writer

import numpy as np
from selectolax.parser import HTMLParser
from tqdm import tqdm

from dst.data import BPEDataset
from dst.data.utils import download_url, split_into_sentences

logger = logging.getLogger(__name__)


class RIADataset(BPEDataset):
    url = 'https://github.com/RossiyaSegodnya/ria_news_dataset/raw/master/ria.json.gz'
    total_size = 1003869
    parts = {'train': -1,
             'dev': 5000,
             'test': 5000}

    def __init__(self,
                 directory: str,
                 prefix: str,
                 part: str,
                 max_sequence_length=150,
                 n_sentences=3,
                 dev=False,
                 seed=42,
                 parts=None,
                 rm_strongs=True,
                 **kwargs):
        self.directory = directory
        self.rm_strongs = rm_strongs

        if dev and part == "train":
            part = "dev"

        if parts is not None:
            RIADataset.parts = parts

        if not self.exist_source(directory):
            self.retrieve_dataset(n_sentences, seed)
        super(RIADataset, self).__init__(directory, prefix, part, max_sequence_length, **kwargs)

    def retrieve_dataset(self, n_sentences, seed):
        temp_dir = tempfile.TemporaryDirectory()
        # temp_dir = namedtuple("Directory", "name")(name=self.directory)
        downloaded_filename = download_url(RIADataset.url, temp_dir.name)
        extracted_filename = downloaded_filename.replace('.gz', '')

        with open(extracted_filename, 'wb') as out_f, gzip.GzipFile(downloaded_filename) as zip_f:
            logger.info("Start extracting")
            # shutil.copyfileobj(zip_f, out_f)
            for data in tqdm(zip_f, total=self.total_size, desc="Extracting"):
                out_f.write(data)
            logger.info("Dataset has been extracted")

        with open(extracted_filename, 'r') as out_f:
            # total_size = self.count_lines(out_f)
            total_size = self.total_size
            train_idx, test_idx = self.get_split_indexes(total_size)

            logger.info("Start splitting dataset into 'train': {} and 'test': {} parts".format(len(train_idx),
                                                                                               len(test_idx)))
            train_filename = os.path.join(self.directory, "train.tsv")
            test_filename = os.path.join(self.directory, "test.tsv")
            dev_filename = os.path.join(self.directory, "dev.tsv")

            with open(train_filename, "w", newline='') as train_file, open(test_filename, "w", newline='') as test_file:
                train_index_cursor, test_index_cursor = 0, 0
                train_writer = writer(train_file, delimiter="\t")
                test_writer = writer(test_file, delimiter="\t")
                dropped = 0

                for i, line in tqdm(enumerate(out_f), total=total_size, desc="Exporting dataset"):
                    data = json.loads(line)
                    title, text = data['title'], data['text']
                    text = self.clear_text(text, self.rm_strongs)
                    text = " ".join(split_into_sentences(text)[:n_sentences])

                    drop = False
                    if len(text) == 0 or len(title) == 0:
                        message = "Found empty article ID: {} Text: {} Title: {}"
                        logger.debug(message.format(i,
                                                    text[:100],
                                                    title[:100]))
                        drop = True
                        dropped += 1

                    if i == train_idx[train_index_cursor]:
                        train_index_cursor += 1
                        if drop:
                            continue
                        train_writer.writerow([text, title])
                    elif i == test_idx[test_index_cursor]:
                        test_index_cursor += 1
                        if drop:
                            continue
                        test_writer.writerow([text, title])
                    else:
                        raise RuntimeError()

            with open(train_filename, "r") as train_file, open(dev_filename, "w") as dev_file:
                for line in itertools.islice(train_file, RIADataset.parts["dev"]):
                    dev_file.write(line)

            logger.info("Exported {} articles, {} empty articles have been dropped".format(i - dropped, dropped))

        temp_dir.cleanup()

    @staticmethod
    def get_split_indexes(total: int, seed: int = None) -> typing.Tuple[np.array]:
        train_size, test_size = RIADataset.parts["train"], RIADataset.parts["test"]
        if train_size == -1:
            train_size = total - test_size
        if test_size == -1:
            test_size = total - train_size

        if train_size + test_size != total:
            raise ValueError("Dataset splits sizes mismatch.")

        if seed is not None:
            np.random.seed(seed)

        random_indexes = np.random.permutation(total)
        train_idx, test_idx = random_indexes[:train_size], random_indexes[-test_size:]
        train_idx, test_idx = map(sorted, [train_idx, test_idx])

        return train_idx, test_idx

    @staticmethod
    def count_lines(file) -> int:
        total = 0
        for line in file.readlines():
            total += 1
        return total

    @staticmethod
    def clear_text(text: str, rm_strong=True) -> str:
        selector = "strong"
        text = unicodedata.normalize("NFKD", text)
        text = text.replace("\n", " ")
        tree = HTMLParser(text)
        if rm_strong:
            for node in tree.css(selector):
                node.decompose()
        return tree.text().strip()

    @staticmethod
    def exist_source(directory):
        existence = [os.path.exists(os.path.join(directory, part + ".tsv")) for part in RIADataset.parts]
        return all(existence)
