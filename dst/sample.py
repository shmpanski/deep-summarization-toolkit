from typing import List, Tuple

import torch

from dst import data, models


class Sampler:
    def __init__(self, model, dataset, inference_args={}, device="cpu"):
        """Model sampler.

        Args:
            model (SummarizationModel): Instatiated model, that have :attr:`inference` method.
            dataset (BPEDataset): Dataset, using to encode/decode sequences into stings.
            inference_args (dict, optional): Defaults to {}. Inference arguments.
            device (str, optional): Defaults to "cpu". Selected device.
        """

        self.model = model
        self.dataset = dataset
        self.inference_args = inference_args
        self.device = device

    def sample(self, input: List[str]) -> List[List[str]]:
        """Sample example.

        Args:
            input (List[str]): Input text, need to summarize.

        Returns:
            List[List[str]]: Summarizations.
        """

        if isinstance(input, str):
            input = [input]
        input = self.dataset.encode(input).to(self.device)
        summaries, _ = self.model.inference(input, **self.inference_args)
        summaries_strs = self.dataset.decode(summaries)

        return summaries_strs

    @staticmethod
    def from_state(model: str,
                   margs: dict,
                   state_path: str,
                   dataset: str,
                   dargs: dict,
                   inference_args={},
                   device="cpu"):
        """Instatiate sampler using dumped model.

        Args:
            model (str): Model name.
            margs (dict): Model arguments.
            state_path (str): Model state file.
            dataset (str): Dataset name.
            dargs (dict): Dataset arguments.
            inference_args (dict, optional): Defaults to {}. Inference arguments.
            device (str, optional): Defaults to "cpu". Used device.

        Returns:
            Sampler: instantiated :classs:`Sampler`.
        """

        dataset = getattr(data, dataset)(part=None, **dargs)
        model, _ = getattr(models, model).create(dataset, margs)
        model.load_state_dict(torch.load(state_path))
        model.to(device)

        return Sampler(model, dataset, inference_args, device)
