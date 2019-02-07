from ignite.metrics import Metric
from rouge import Rouge


class RougeMetric(Metric):
    def __init__(self, output_transform=lambda x: x, batch_size=lambda x: len(x), **kwargs):
        self._stats, self._metrics = kwargs["stats"], kwargs["metrics"]
        self._batch_size = batch_size
        self._count = 0
        self._total_stats = {}
        super(RougeMetric, self).__init__(output_transform)
        self.rouge = Rouge(**kwargs)

    def update(self, output):
        self._count += 1
        try:
            rouge_res = self.rouge.get_scores(output[0], output[1], avg=True)
            for metric, metric_val in rouge_res.items():
                for stat, val in metric_val.items():
                    self._total_stats[metric][stat] += val
        except ValueError:
            return

    def reset(self):
        self._total_stats = {metric: {stat: 0 for stat in self._stats} for metric in self._metrics}
        self._count = 0

    def compute(self):
        for metric, metric_val in self._total_stats.items():
            for stat, val in metric_val.items():
                self._total_stats[metric][stat] /= (self._count)
        return self._total_stats
