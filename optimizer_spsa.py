import numpy as np
from matplotlib import pyplot as plt
from simulation import Simulation, IMPOptMea
from optimizer_fdsa import OptimizeTotalCostFDSA
from scipy.stats import bernoulli


class OptimizeTotalCostSPSA(Simulation):
    imp_opt_mea = IMPOptMea
    print_res = None

    def __init__(self, p1, p2, p3, p4):
        Simulation.__init__(self, p1, p2, p3, p4)
        self.regen_hist = []
        self.sim_init()

    def spsa(self,max_measure=100):
        self.spsa_seq = []
        self.top_measure = 0
        self.sim_measure = 0
        self.min_cost_measure = self.top_measure
        self.min_cost = None
        self.min_cost_im_min_s = None
        self.min_cost_im_max_s = None
        self.spsa_dim = 2
        self.cotd_threshold = 0.9
        self.spsa_latest_best = 0.0
        self.spsa_latest_best_measure = 0
        self.opt_hist_tc = []
        self.opt_hist_min_s = []
        self.opt_hist_max_s = []

        while self.top_measure <= max_measure:
            self.opt_ss_spsa()
            if self.top_measure == 1:
                self.init_cost = self.model.cum_cost
                self.min_cost = self.init_cost
                self.min_cost_measure = self.top_measure

            if self.spsa_res == 'S1':
                self.spsa_latest_best = self.model.cum_cost
                self.spsa_latest_best_measure = self.top_measure
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

        opt_plot = plt
        opt_plot.plot(self.opt_hist_tc)

    def opt_ss_spsa(self):
        self.spsa_res = ""
        yk = self.sim_sc1("y(Theta)")
        cur_theta1 = self.model.im_min_s - self.model.min_s_lower_bound
        cur_theta2 = self.model.im_max_s - self.model.im_min_s
        cur_imp = {"im_min_s": self.model.im_min_s, "im_max_s": self.model.im_max_s}
        ak = spsa_gain_seq_ak(self.top_measure)
        ck = spsa_gain_seq_ck(self.top_measure)
        delta_k = spsa_delta_k(self.top_measure)
        delta_kp = spsa_delta_kp(self.spsa_dim)
        gradient = self.__estimate_gradient(ck, delta_k, delta_kp)
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
            self.spsa_res = 'F1'
            return

        if yk_new["cum_otd"] < self.cotd_threshold and cost_change < 0:
            print(f"F2:New Theta has the Cumulative OTD rate less than threshold, but cost decreased~")
            self.model.reset_im_policy(cur_imp)
            self.spsa_res = 'F2'
            return

        if cost_change > 0:
            print(f"F3: cost increased  {cost_change} gradient fail!")
            self.model.reset_im_policy(cur_imp)
            self.spsa_res = 'F3'
            return
        if cost_change == 0:
            print(f"S0: cost  not changed {cost_change} gradient staged!")
            self.model.reset_im_policy(cur_imp)
            self.spsa_res = 'F3'
            return
        else:
            print(f"cost reduced {cost_change} gradient work!")
            self.spsa_res = 'S1'

        self.spsa_seq.append({"iter": self.top_measure, "ak": ak, "ck": ck, "delta_k": delta_k, "delta_k": delta_kp
                              , "delta_theta": ak * gradient["im_min_s_gd"], "new_theta1": new_theta1,
                              "new_theta2": new_theta2, "gradient":gradient, "min_s": self.model.im_min_s,
                              "max_s": self.model.im_max_s, "cost_change":cost_change, "spsa_res": self.spsa_res})

    def __estimate_gradient(self, ck=1, delta_k=1, delta_kp=(1,-1)):

        theta1 = self.model.im_min_s - self.model.min_s_lower_bound
        theta2 = self.model.im_max_s - self.model.im_min_s

        y1_theta = (theta1 + ck * delta_k, theta2 + ck * delta_k)
        y2_theta = (theta1 - ck * delta_k, theta2 - ck * delta_k)

        min_s_max_s_set0 = {"im_min_s": self.model.im_min_s, "im_max_s": self.model.im_max_s}
        min_s_max_s_set1 = {"im_min_s": self.model.min_s_lower_bound + y1_theta[0], "im_max_s": self.model.im_max_s + y1_theta[1]}
        min_s_max_s_set2 = {"im_min_s": self.model.min_s_lower_bound - y2_theta[0], "im_max_s": self.model.im_max_s - y2_theta[1]}\

        self.model.reset_im_policy(min_s_max_s_set1)
        y1 = self.sim_sc1("y1(Theta+ck*delta_k)")
        self.model.reset_im_policy(min_s_max_s_set2)
        y2 = self.sim_sc1("y2(Theta-ck*delta_k)")
        delta_y = y1["cum_cost"] - y2["cum_cost"]
        gd_min_s = delta_y / (2*ck*delta_kp[0])
        gd_max_s = delta_y / (2*ck*delta_kp[1])
        self.model.reset_im_policy(min_s_max_s_set0)   # set back to original im policy
        return {"im_min_s_gd": gd_min_s, "im_max_s_gd": gd_max_s}


def convert0(n):
    if n == 0:
        n = -1 * 250
    else:
        n = 1*250

    return n


def spsa_delta_k(k):
    return 1


def spsa_delta_kp(dim):
    delta_kp = map(convert0, bernoulli.rvs(p=0.5, size=dim))
    return list(delta_kp)


def spsa_gain_seq_ck(k,c=1,gamma=0.101):  # given k, return ck
    ck = c/(k+1)**gamma
    return ck


def spsa_gain_seq_ak(k,a=1, A=1, alpha=0.602 ):  # given k, return ak
    ak = a/((k+1+A)**alpha)
    return ak





