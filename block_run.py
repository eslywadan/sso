from simulation import *
from controlled_random import ControlRandomNumber as crn
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt



class BlockRun:

    def __init__(self, ctl_ran:crn):
        self.ctl_ran = ctl_ran
        self.run_cost = []
        self.avg_cost = 0
        self.sum_cost = 0
        self.run_otd = []
        self.avg_otd = 0
        self.sum_otd = 0
        self.run_fulfill_demand = []
        self.sum_fulfill_demand = 0
        self.avg_fulfill_demand = 0

    def block_avg(self, sim_model:Simulation, shuffle=True, viz=False):
        if shuffle is True:
            np.random.shuffle(self.ctl_ran.seed_seq)
        while self.ctl_ran.cur_seed_ix < self.ctl_ran.seed_num-1:
            sim_model.sim_simple(self.ctl_ran.cur_seed)

            self.run_cost.append(sim_model.model.cum_cost)
            self.sum_cost += sim_model.model.cum_cost

            self.run_otd.append(sim_model.model.cum_otd)
            self.sum_otd += sim_model.model.cum_otd

            self.run_fulfill_demand.append(sim_model.model.cum_fulfill_qty)
            self.sum_fulfill_demand += sim_model.model.cum_fulfill_qty
            self.ctl_ran.next_as_cur_ix()

        self.ctl_ran.next_as_cur_ix()
        self.avg_cost = self.sum_cost/self.ctl_ran.seed_num
        self.avg_otd = self.sum_otd/self.ctl_ran.seed_num
        self.avg_fulfill_demand = self.sum_fulfill_demand/self.ctl_ran.seed_num

        if viz:
            self.viz_res()

    def viz_res(self):
        self.res_hist()

    def res_hist(self):
        sns.histplot(self.run_cost)
        sns.histplot(self.run_otd)
        sns.histplot(self.run_fulfill_demand)


if __name__ == "__main__":

    breakpoint()
