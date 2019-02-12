import unittest
import tempfile
import os

from dst.data import BPEDataset


class TestBPEDatasetMethods(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.prefix = "test"
        self.part = "train"
        self.dev_part = "dev"
        self.workdir = os.path.join(self.directory.name, self.prefix)

        train_source_file_name = os.path.join(self.directory.name, "train.tsv")
        with open(train_source_file_name, "w+") as train_file:
            train_file.writelines(["I have writen some code\tWrite code\r\n",
                                   "There is an apple on the table\tApple on the table\r\n",
                                   "I like cats and dogs\tCats and dogs\r\n"])

        dev_source_file_name = os.path.join(self.directory.name, "dev.tsv")
        with open(dev_source_file_name, "w+") as dev_file:
            dev_file.writelines(["I have writen some code\tWrite code\r\n",
                                 "There is an apple on the table\tApple on the table\r\n",
                                 "I like cats and dogs\tCats and dogs\r\n",
                                 "There is an apple on the table\tApple on the table\r\n",
                                 "I like cats and dogs\tCats and dogs\r\n"])

    def tearDown(self):
        self.directory.cleanup()

    def test_preprocess(self):
        train_dataset = BPEDataset(self.directory.name, self.prefix, self.part,
                                   vocab_size=30, embedding_size=64)
        train_data_file = os.path.join(self.workdir, "train.npy")
        spm_data_file = os.path.join(self.workdir, "spm.model")

        self.assertEqual(len(train_dataset), 3)
        self.assertTrue(os.path.exists(train_data_file))
        self.assertTrue(os.path.exists(spm_data_file))

    def test_exist(self):
        exist_before_creating = BPEDataset.exist(self.directory.name, self.prefix, "train")
        train_dataset = BPEDataset(self.directory.name, self.prefix, self.part,
                                   vocab_size=30, embedding_size=64)
        exist_after_creating = BPEDataset.exist(self.directory.name, self.prefix, "train")
        self.assertFalse(exist_before_creating)
        self.assertTrue(exist_after_creating)

    def test_collate_function(self):
        train_dataset = BPEDataset(self.directory.name, self.prefix, self.part,
                                   vocab_size=30, embedding_size=64)
        batch = [train_dataset[i] for i in range(3)]
        batch = train_dataset.collate_function(batch)
        self.assertEqual(len(batch['src_length']), 3)
        self.assertEqual(len(batch['trg_length']), 3)
        self.assertEqual(batch['src'].shape, (3, max(batch['src_length'])))
        self.assertEqual(batch['trg'].shape, (3, max(batch['trg_length'])))

    def test_init_multiple_parts(self):
        train_dataset = BPEDataset(self.directory.name, self.prefix, self.part,
                                   vocab_size=30, embedding_size=64)
        dev_dataset = BPEDataset(self.directory.name, self.prefix, self.dev_part,
                                 spm=train_dataset.spm)
        exist = BPEDataset.exist(self.directory.name, self.prefix, "dev")
        self.assertEqual(len(dev_dataset), 5)
        self.assertTrue(exist)

    def test_get_embeddings(self):
        dataset = BPEDataset(self.directory.name, self.prefix, self.part,
                             vocab_size=30, embedding_size=64)
        embeddings = dataset.get_embeddings()
        self.assertTupleEqual(embeddings.shape, (30, 64))
