import string
import numpy as np
from inventory_model import InvMdlSingle, GenLeadTime, GenDemand
from visualize import VisualizeModel


class Simulation:
    """The Simulation Class (1) feasible solution
                            (2) Optimal solution """
    sim_measure = None
    sim_hist = None

    def __init__(self, p1, p2, p3, p4):
        self.model = InvMdlSingle(p1, p2, p3, p4)

    def sim_init(self):
        """Init the simulation """
        self.model.gen_demand()
        self.model.gen_lt()
        self.model.reset_plan()
        self.model.init_ss()
        Scatter2Dims.plot_dist_scatter(self.model.demand, self.model.order_lt)

    def sim_sc1(self, pid: string) -> dict:
        """Simulate Scenario 1 approach re-plan by keeping demand and LT   """
        self.model.reset_plan()
        self.model.bal_all()
        self.sim_measure += 1
        self.sim_hist.append(self.model.cum_cost)
        return self.outline_plan_res(pid)

    def sim_sc2(self):
        """Simulate Scenario 2 approach re-plan by keeping demand and regen LT   """
        self.model.gen_lt()
        self.model.reset_plan()
        self.model.bal_all()
        self.outline_plan_res("")

    def sim_sc3(self):
        """Simulate Scenario 3 approach re-plan by regen demand while keep LT   """
        self.model.gen_demand()
        self.model.reset_plan()
        self.model.bal_all()
        self.outline_plan_res("")

    def outline_plan_res(self, pid):
        if not self.print_res:
            pass
        else:
            print(f"Cumulative On Time Delivery rate: {self.model.cum_otd}\n"
              f"Cumulative Demand Total: {sum(self.model.demand)}\n "
              f"Cumulative Fulfill Total:{self.model.cum_fulfill_qty}\n"
              f"Cumulative Dismissed Total: {self.model.cum_nsfs_qty}\n"
              f"s/S : min_s : {self.model.im_min_s} | max_S : {self.model.im_max_s} \n"
              f"Total Cost: {round(self.model.cum_cost)}\n"
              f"PO Cost: {round(self.model.cum_po_cost)}\n"
              f"Holding Cost: {round(self.model.cum_holding_cost)}\n")

        return dict(plan_id=pid, im_min_s=self.model.im_min_s,
                    im_max_s=self.model.im_max_s, cum_cost=self.model.cum_cost,
                    cum_otd=self.model.cum_otd)


class Scatter2Dims:

    @staticmethod
    def plot_normal_scatter():
        d3 = np.random.normal(6, 6, 1000)
        d4 = np.random.normal(100, 100, 1000)
        x = {"Label": 'Demand', "Data": d3}
        y = {"Label": 'Lead Time', "Data": d4}
        p2 = VisualizeModel(x, y)
        p2.plot_scatter_hist()

    @staticmethod
    def plot_stochastic_demand_lt_scatter(lam=6, mu=100, sample=1000):
        """   """
        d1 = GenLeadTime.gen_lead(lam, sample)
        d2 = GenDemand.gen_demand(mu, sample)
        x = {"Label": 'Demand', "Data": d1}
        y = {"Label": 'Lead Time', "Data": d2}
        p1 = VisualizeModel(x, y)
        p1.plot_scatter_hist()

    @staticmethod
    def plot_dist_scatter(d1, d2):
        x = {"Label": '', "Data": np.array(d1)}
        y = {"Label": '', "Data": np.array(d2)}
        p1 = VisualizeModel(x, y)
        p1.plot_scatter_hist()

