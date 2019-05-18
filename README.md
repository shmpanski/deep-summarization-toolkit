# Deep Summarization Toolkit

Simple toolkit for deep summarization model training, evalutaion and sampling.

- [Deep Summarization Toolkit](#deep-summarization-toolkit)
  - [Toolkit description](#toolkit-description)
    - [Requirements](#requirements)
    - [Installation](#installation)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Sampling](#sampling)
  - [Examples](#examples)
  - [TODOs](#todos)

## Toolkit description

This toolkit is used for simplify model training/testing/sampling.
It contains a few summarization datasets and models.
You can easily extend this toolkit with your own models and dataset.

We use `json` or `yaml` configuration files in order to simplify training and testing multiple models.
This files should contain model description, training, evaluating and sampling specificity.
Also, we would like configurations to be flexible and simple.
See [workbench](./workbench) readme for more details and datasets and models overview.

### Requirements

- python >= 3.5
- pytorch >= 1.1.0
- gensim

Other dependencies can be found in file [requirements.txt](./requirements.txt).

### Installation

```bash
git clone https://github.com/gooppe/deep-summarization-toolkit.git
cd deep-summarization-toolkit
pip3 install -r requirements.txt
```

### Training

Training is simple. Just describe you run configuration in `.yml` or `.json` file or use existing configurations, for example dev run file for rnn model:

```bash
python3 train.py workbench/dev-ria-20-rnn.yml
```

Toolkit cannot working in daemon mode, that's why we recommend to use `tmux` or `nohup` utils for running model.

### Evaluation

You can run evaluation using run configuration and pretrained model:

```bash
python3 eval.py my_config_file.yml my_pretrained_mode.pth
```

### Sampling

Sampling is similar to validation except for additional input and output files with text and summarization respectively.

```bash
python3 sample.py my_config_file.yml my_pretrained_mode.pth input.txt output.txt
```

## Examples

You can find some examples in [workbench](/workbench) directory.

## TODOs

- Add more models
- Improve beam search perfomance
- Improve transformer perfomance
- Make toolkit portable
- Customize metrics
