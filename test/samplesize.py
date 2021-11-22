import numpy as np
import math
import visualize


class SampleSize:
    sample_size: list[float]
    portion: list[float]

    def __init__(self,sim_size,rng):
        self.limit = rng
        self.gen_np(sim_size)

    def gen_np(self,sim_size):
        self._gen_n(sim_size)

    def _gen_p(self, num):
        low_rng = 0.0 + self.limit
        high_rng = 1.0 - self.limit
        p_sample = np.random.uniform(low=low_rng, high=high_rng, size=num)
        self.portion = p_sample

    def _gen_n(self, sim_num):
        self._gen_p(sim_num)
        self.sample_size = []
        for p in self.portion:
            if p == 0 or p == 1:
                n = float('inf')
            if p <= 0.50:
                n = 10.0/p
            if p > 0.50:
                n = 10.0/(1-p)
            self.sample_size.append(math.ceil(n))

    def plot_np(self):
        x = {"Data": np.array(self.portion), "Label": "Portion"}
        y = {"Data": np.array(self.sample_size), "Label": "Sample Size"}
        visualize.VisualizeModel(x, y)
        p1 = visualize.VisualizeModel(x, y)
        p1.plot_scatter()


try_sample = SampleSize(1000,0.05)
try_sample.plot_np()