# Workbench description

Use `.json` or `.yaml` file to describe your run configuration.
This files are validated during toolkit running using [schema-file](../dst/schemas/).
This schema also contains descriptions and default configuration values.
For clarity, look at the examples above.
Dev-prefixed files contain lightweighted model configurations. 

- [Workbench description](#workbench-description)
  - [Available models](#available-models)
    - [Attentional RNN](#attentional-rnn)
    - [Vanilla transformer](#vanilla-transformer)
    - [Phrase-based attentional transformer](#phrase-based-attentional-transformer)
  - [Available datasets](#available-datasets)
    - [Byte-pair-encoding dataset](#byte-pair-encoding-dataset)
    - [RIA dataset](#ria-dataset)

## Available models

Models should stored in `dst.models` module.
Pipeline searches models in this module during model instantiation.
All this models supports slow beam-search decoding.

### Attentional RNN

This is simple recurrent based attentional model. See [source](../dst/models/rnn.py) for more details.

### Vanilla transformer

This simple transformer from [original paper](https://arxiv.org/abs/1706.03762).
See [source](../dst/models/transformer.py) for more detals.

### Phrase-based attentional transformer

This is transformer with modificated [phrase-based attention](https://arxiv.org/abs/1810.03444).
See [source](../dst/models/pba_transformer.py) for more details.

## Available datasets

Models should stored in `dst.data` module.
Pipeline searches models in this module during dataset instantiation.
As baseline, datasets supports bpe encoding only.

### Byte-pair-encoding dataset

This is base dataset. Use `train.tsv`, `test.tsv` and `dev.tsv` files with data. All this files must be stored in some dataset folder and contain summarization examples, separated by `\t` tabulation symbol. For example:
```
This is a good example of train instance. \tThis is summarization of a good train example instance.
```
where first sentence is original text, last sentence is it's summarization.
This dataset uses [sentencepiece](https://github.com/google/sentencepiece) processor to build vocabulary and train bpe tokenization. Also, its possible to automatically pretrain gensim word2vec embeddings. See [source](../dst/data/bpe_dataset.py) for more detail.

### RIA dataset

This dataset based on byte-pair-encoding dataset. It automaticaly download, preprocess and uses RIA-news corpus as dataset. See [source](../dst/data/ria_dataset.py) for more detail.