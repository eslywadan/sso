from simulation import Simulation, Scatter2Dims
import numpy as np
from matplotlib import pyplot as plt


class IMPOptMea:
    """Record Set Store the Optimizer Hist for Inv Man Policy (s,S)"""
    pid = None

    def __init__(self, **kwargs):
        self.measure = []
        self.theta1 = []
        self.theta2 = []
        self.min_s = []
        self.max_s = []
        self.demand_total = []
        self.demand_fulfill = []
        self.back_order_total = []
        self.total_cost = []
        self.total_holding_cost = []
        self.total_po_cost = []
        self.cum_otd = []

    def append_rec(self, **kwargs):
        pass


class OptimizeTotalCostFDSA(Simulation):

    imp_opt_mea = IMPOptMea

    def __init__(self, p1, p2, p3, p4):
        Simulation.__init__(self, p1, p2, p3, p4)
        self.regen_hist = []
        self.sim_init()

    def fdsa(self, ak=1, ck=10, kasi_min_s=1, kasi_max_s=1, cotd_threshold=0.9, max_measure=1000,
             use_the_last_best=False):
        self.top_measure = 0
        self.sim_measure = 0
        self.fdsa_ak = ak
        self.fdsa_ck = ck
        self.fdsa_kasi1 = kasi_min_s
        self.fdsa_kasi2 = kasi_max_s
        self.opt_hist_tc = []
        self.opt_hist_min_s = []
        self.opt_hist_max_s = []
        self.sim_hist = []
        self.min_cost = 0
        self.model.init_ss(int(np.random.uniform(1000, 3000, 1)), use_the_last_best=use_the_last_best)
        self.print_res = False
        while self.top_measure <= max_measure:
            ak = self.fdsa_ak
            ck = self.fdsa_ck
            kasi_min_s = self.fdsa_kasi1
            kasi_max_s = self.fdsa_kasi2
            self.opt_ss_with_fixed_dem_lt(ak, ck, kasi_min_s, kasi_max_s, cotd_threshold)
            print(self.fdsa_res)
            if self.fdsa_res != 'S1':
                self.fdsa_tune_para()

            if self.top_measure == 1:
                self.init_cost = self.model.cum_cost
                self.min_cost = self.init_cost
                self.min_cost_measure = self.top_measure

            if self.fdsa_res == 'S1':
                self.fdsa_latest_best = self.model.cum_cost
                self.fdsa_latest_best_measure = self.top_measure
                if self.min_cost is None:
                    self.min_cost = self.model.cum_cost
                if self.model.cum_cost < self.min_cost:
                    self.min_cost = self.model.cum_cost
                    self.min_cost_measure = self.top_measure
                    self.min_cost_im_min_s = self.model.im_min_s
                    self.min_cost_im_max_s = self.model.im_max_s

            self.opt_hist_tc.append(self.model.cum_cost)
            self.opt_hist_min_s.append(self.model.im_min_s)
            self.opt_hist_max_s.append(self.model.im_max_s)
            self.top_measure += 1

        print(f"Top Measure : {self.top_measure}\n"
              f"Detail Measure : {self.min_cost_measure}\n"
              f"Initial Cost: {self.init_cost}\n"
              f"Minimum Cost: {self.min_cost}\n"
              f"Minimum Cost Measure: {self.min_cost_measure}")
        opt_plot = plt
        opt_plot.plot(self.opt_hist_tc)
        # sim_plot = plt
        # sim_plot.plot(self.sim_hist)

    def update_opt_hist_record(self):

        self.opt_hist_record.append_rec()

    def fdsa_tune_para(self):
        if self.fdsa_res == 'F1':
            self.fdsa_ak = (self.fdsa_ak - 0.2*self.fdsa_ak) / self.fdsa_ck

        if self.fdsa_res == 'F2':
            self.fdsa_ak = (self.fdsa_ak - 0.2*self.fdsa_ak) / self.fdsa_ck

        if self.fdsa_res == 'F3':
            self.fdsa_ck = (self.fdsa_ck + 0.1*self.fdsa_ck) / self.fdsa_ak

        if self.fdsa_res == 'S0':
            self.fdsa_ck = (self.fdsa_ck - 0.1*self.fdsa_ck) / self.fdsa_ak
            self.fdsa_ak = self.fdsa_ak + 0.1*self.fdsa_ak / self.fdsa_ck

    def opt_ss_with_fixed_dem_lt(self, ak=1, ck=1, kasi_min_s=1, kasi_max_s=1, cotd_threshold=0.9):

        """ theta1 is the min_s - min_lower_bound and theta2 here is the delta_s, the max_s = theta1 + theta2"""
        self.cotd_threshold = cotd_threshold
        self.fdsa_res = ""
        yk = self.sim_sc1("y(Theta_k)")
        cur_theta1 = self.model.im_min_s - self.model.min_s_lower_bound
        cur_theta2 = self.model.im_max_s - self.model.im_min_s
        cur_imp = {"im_min_s": self.model.im_min_s, "im_max_s": self.model.im_max_s}
        gradient = self.__estimate_gradient(ck, kasi_min_s, kasi_max_s)
        print(gradient)

        new_theta1 = cur_theta1 - ak * gradient["im_min_s_gd"]
        new_theta2 = cur_theta2 - ak * gradient["im_max_s_gd"]

        if new_theta1 < 0:
            new_theta1 = 0
        if new_theta2 < 0:
            new_theta2 = 0
        new_min_s = self.model.min_s_lower_bound + new_theta1
        new_max_s = new_min_s + new_theta2

        new_imp = {"im_min_s": new_min_s, "im_max_s": new_max_s}
        self.model.reset_im_policy(new_imp)
        yk_new = self.sim_sc1("y(Theta_k+1")

        cost_change = yk_new["cum_cost"] - yk["cum_cost"]

        if yk_new["cum_otd"] < self.cotd_threshold and cost_change > 0:
            print(f"F1:New Theta has the Cumulative OTD rate less than threshold, sol fail~")
            self.model.reset_im_policy(cur_imp)
            self.fdsa_res = 'F1'
            return

        if yk_new["cum_otd"] < self.cotd_threshold and cost_change < 0:
            print(f"F2:New Theta has the Cumulative OTD rate less than threshold, but cost decreased~")
            self.model.reset_im_policy(cur_imp)
            self.fdsa_res = 'F2'
            return

        if cost_change > 0:
            print(f"F3: cost increased  {cost_change} gradient fail!")
            self.model.reset_im_policy(cur_imp)
            self.fdsa_res = 'F3'
            return
        if cost_change == 0:
            print(f"S0: cost  not changed {cost_change} gradient staged!")
            self.model.reset_im_policy(cur_imp)
            self.fdsa_res = 'F3'
            return
        else:
            print(f"cost reduced {cost_change} gradient work!")
            self.fdsa_res = 'S1'

    def __estimate_gradient(self, ck=1, kasi_min_s=1, kasi_max_s=1):

        theta1 = self.model.im_min_s - self.model.min_s_lower_bound
        theta2 = self.model.im_max_s - self.model.im_min_s

        plus_fd_min_s = theta1 + ck * kasi_min_s
        minus_fd_min_s = theta1 - ck * kasi_min_s
        plus_fd_max_s = theta2 + ck * kasi_max_s
        minus_fd_max_s = theta2 - ck * kasi_max_s

        min_s_max_s_set0 = {"im_min_s": self.model.im_min_s, "im_max_s": self.model.im_max_s}
        min_s_max_s_set1 = {"im_min_s": self.model.min_s_lower_bound + plus_fd_min_s, "im_max_s": self.model.im_max_s}
        min_s_max_s_set2 = {"im_min_s": self.model.min_s_lower_bound + minus_fd_min_s, "im_max_s": self.model.im_max_s}
        min_s_max_s_set3 = {"im_min_s": self.model.im_min_s, "im_max_s": self.model.im_min_s + plus_fd_max_s}
        min_s_max_s_set4 = {"im_min_s": self.model.im_min_s, "im_max_s": self.model.im_min_s + minus_fd_max_s}

        self.model.reset_im_policy(min_s_max_s_set1)
        y1p = self.sim_sc1("y(Theta+ck*kasi1)")
        self.model.reset_im_policy(min_s_max_s_set2)
        y1n = self.sim_sc1("y(Theta-ck*kasi1)")
        gd_min_s = (y1p["cum_cost"] - y1n["cum_cost"]) / (2*ck)
        self.model.reset_im_policy(min_s_max_s_set3)
        y2p = self.sim_sc1("y(Theta+ck*kasi2)")
        self.model.reset_im_policy(min_s_max_s_set4)
        y2n = self.sim_sc1("y(Theta-ck*kasi2)")
        gd_max_s = (y2p["cum_cost"] - y2n["cum_cost"]) / (2*ck)
        self.model.reset_im_policy(min_s_max_s_set0)   # set back to original im policy
        return {"im_min_s_gd": gd_min_s, "im_max_s_gd": gd_max_s}
