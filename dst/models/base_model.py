from torch import nn


class BaseSummarizationModel(nn.Module):
    def __init__(self):
        super(BaseSummarizationModel, self).__init__()

    def forward(self, source, target, **kwargs):
        raise NotImplementedError

    def inference(self, source, limit, **kwargs):
        raise NotImplementedError

    def create_trainer(self, optimizer, device):
        raise NotImplementedError

    def create_evaluator(self, device):
        raise NotImplementedError

    @staticmethod
    def create(dataset, margs):
        raise NotImplementedError

    def learnable_parameters(self):
        """Get all learnable parameters of the model.

        Returns: Generator of parameters.

        """
        for param in self.parameters():
            if param.requires_grad:
                yield param
