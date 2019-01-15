import unittest
import tempfile
import os

from data import BPEDataset


class TestBPEDatasetMethods(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.prefix = "test"
        self.parts = ["train"]
        self.workdir = os.path.join(self.directory.name, self.prefix)

        train_source_file_name = os.path.join(self.directory.name, "train.tsv")
        with open(train_source_file_name, "w+") as train_file:
            train_file.writelines(["I have writen some code\tWrite code\r\n",
                                   "There is an apple on the table\tApple on the table\r\n",
                                   "I like cats and dogs\tCats and dogs\r\n"])

    def tearDown(self):
        self.directory.cleanup()

    def test_preprocess(self):
        train_dataset, = BPEDataset.preprocess(self.directory.name, self.prefix, self.parts,
                                               vocab_size=30, embedding_size=64)
        train_data_file = os.path.join(self.workdir, "train.npy")
        spm_data_file = os.path.join(self.workdir, "spm.model")

        self.assertEqual(len(train_dataset), 3)
        self.assertTrue(os.path.exists(train_data_file))
        self.assertTrue(os.path.exists(spm_data_file))

    def test_exist(self):
        exist_before_creating = BPEDataset.exist(self.directory.name, self.prefix, "train")
        train_dataset, = BPEDataset.preprocess(self.directory.name, self.prefix, self.parts,
                                               vocab_size=30, embedding_size=64)
        exist_after_creating = BPEDataset.exist(self.directory.name, self.prefix, "train")
        self.assertFalse(exist_before_creating)
        self.assertTrue(exist_after_creating)

    def test_collate_function(self):
        train_dataset, = BPEDataset.preprocess(self.directory.name, self.prefix, self.parts,
                                               vocab_size=30, embedding_size=64)
        batch = [train_dataset[i] for i in range(3)]
        batch = train_dataset.collate_function(batch)
        self.assertEqual(len(batch['src_length']), 3)
        self.assertEqual(len(batch['trg_length']), 3)
        self.assertEqual(batch['src'].shape, (3, max(batch['src_length'])))
        self.assertEqual(batch['trg'].shape, (3, max(batch['trg_length'])))
