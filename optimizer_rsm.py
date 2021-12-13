import numpy as np
import pandas as pd
import patsy as pt

"""Reference: https://towardsdatascience.com/a-simple-guide-to-linear-regression-using-python-7050e8c751c1 """
import statsmodels.regression.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns


from controlled_random import ControlRandomNumber as crn
from simulation import Simulation
from block_run import BlockRun
from main import PlanEnv as plan_env


class ExpRec:
    exp: int
    theta1: int
    theta2: int
    min_s: int
    max_s: int
    avg_cost: float
    avg_otd: float
    avg_fulfill: int
    avg_unit_cost: float
    avg_period_cost: float


class OtdDelta:
    theta1: float
    theta2: float

    def __init__(self):
        self.theta1 = 0
        self.theta2 = 0


class Experiment:

    def __init__(self):
        self.otd_delta = OtdDelta()
        self.cost_delta = OtdDelta()
        self.block_ctl_rand = crn(100)
        self.hist_exog_vars = []
        self.hist_endog_cost = []
        self.hist_endog_otd = []
        self.hist_endog_fulfill = []
        self.reset_exp_res()
        self.sim_model = Simulation(p1=plan_env.env_set, p2=plan_env.cost_set, p3=plan_env.stochastic_set,
                                    p4=plan_env.min_s_max_s_set)

    def reset_exp_res(self):
        self.exp_exog_vars = []
        self.exp_endog_cost = []
        self.exp_endog_otd = []
        self.exp_endog_fulfill = []


class Exp2F2L(Experiment):
    theta1:float
    theta2:float

    def find_feasible_center(self, block_run=5, init_theta1=100, init_theta2=600, otd_threshold=0.80, min_cost=False):
        self.block_run(block_run, init_theta1, init_theta2)
        self.exp_res_avg_otd = sum(self.exp_endog_otd)/len(self.exp_endog_otd)
        self.exp_res_avg_cost = sum(self.exp_endog_cost)/len(self.exp_endog_cost)
        self.otd_threshold = otd_threshold

        iter_num, max_iter = 0, 100
        while self.exp_res_avg_otd < self.otd_threshold and iter_num <= max_iter :
            self.exp_res_avg_otd_prev = self.exp_res_avg_otd
            self.exp_res_avg_cost_prev = self.exp_res_avg_cost
            self.check_theta_delta(min_cost)
            theta1 = self.theta1 + self.otd_delta.theta1 + self.cost_delta.theta1
            theta2 = self.theta2 + self.otd_delta.theta2 + self.cost_delta.theta2
            self.reset_exp_res()
            self.block_run(block_run,theta1,theta2)
            self.exp_res_avg_otd = sum(self.exp_endog_otd) / len(self.exp_endog_otd)
            self.exp_res_avg_cost = sum(self.exp_endog_cost) / len(self.exp_endog_cost)
            # print(f"average otd: {self.exp_res_avg_otd}")
            iter_num += 1

        self.exp_res_avg_cost = sum(self.exp_endog_cost) / len(self.exp_endog_cost)
        self.exp_res_avg_fulfill_demand = sum(self.exp_endog_fulfill) / len(self.exp_endog_fulfill)
        print(f"Iter num: {iter_num}")
        print(f"Found feasible theta1: {self.theta1} theta2:{self.theta2} "
              f"s1: {self.sim_model.model.im_min_s} s2: {self.sim_model.model.im_max_s}"
              f"avg cost: {self.exp_res_avg_cost}"
              f"avg otd: {self.exp_res_avg_otd}"
              f"avg fulfill demand: {self.exp_res_avg_fulfill_demand}")

    def check_theta_delta(self, min_cost):
        if min_cost is False:
            self.get_otd_delta()
            self.cost_delta.theta1 = 0
            self.cost_delta.theta2 = 0
        else:
            self.get_cost_delta()
            self.otd_delta.theta1 = 0
            self.otd_delta.theta2 = 0

    def get_otd_delta(self):
        self.fit_hist_otd_ols()
        # print(self.fit_ols_res.summary())
        if self.fit_otd_ols_res.rsquared >= 0.6:
            self.otd_delta.theta1 = 10 * self.fit_otd_ols_res.params[1]/abs(self.fit_otd_ols_res.params[2])
            self.otd_delta.theta2 = 10 * self.fit_otd_ols_res.params[2] / abs(self.fit_otd_ols_res.params[1])
        else:
            if self.exp_res_avg_otd_prev < self.exp_res_avg_otd:
                self.otd_delta.theta1 = 30
                self.otd_delta.theta2 = 20
            else:
                self.otd_delta.theta1 = 10
                self.otd_delta.theta2 = 20

    def fit_hist_otd_ols(self):
        X = np.column_stack(self.hist_exog_vars)
        X = np.column_stack(X)
        y = np.array(self.hist_endog_otd)
        rsm = RSModel(X, y)
        rsm.fit_ols()
        self.fit_otd_ols_res = rsm.fit_ols_res
        self.rsm = rsm

    def find_min_cost(self, block_run=5, theta1=100, theta2=200, otd_threshold=0.80, min_cost=True):
        self.block_run(block_run, theta1=theta1, theta2=theta2)
        self.exp_res_avg_otd = sum(self.exp_endog_otd)/len(self.exp_endog_otd)
        self.exp_res_avg_cost = sum(self.exp_endog_cost)/len(self.exp_endog_cost)
        self.exp_best_avg_cost = self.exp_res_avg_cost
        self.otd_threshold = otd_threshold

        iter_num, max_iter = 0, 30
        while iter_num <= max_iter and self.exp_best_avg_cost <= self.exp_res_avg_cost:
            self.exp_res_avg_otd_prev = self.exp_res_avg_otd
            self.exp_res_avg_cost_prev = self.exp_res_avg_cost
            self.check_theta_delta(min_cost)
            theta1 = self.theta1 + self.cost_delta.theta1
            theta2 = self.theta2 +  self.cost_delta.theta2
            self.reset_exp_res()
            self.block_run(block_run,theta1,theta2)
            self.exp_res_avg_otd = sum(self.exp_endog_otd) / len(self.exp_endog_otd)
            self.exp_res_avg_cost = sum(self.exp_endog_cost) / len(self.exp_endog_cost)
            if self.exp_res_avg_cost < self.exp_best_avg_cost and self.exp_res_avg_otd < self.otd_threshold:
                self.exp_best_avg_cost = self.exp_res_avg_cost
                return
            # print(f"average otd: {self.exp_res_avg_otd}")
            iter_num += 1

        self.exp_res_avg_cost = sum(self.exp_endog_cost) / len(self.exp_endog_cost)
        self.exp_res_avg_fulfill_demand = sum(self.exp_endog_fulfill) / len(self.exp_endog_fulfill)
        print(f"Iter num: {iter_num}")
        print(f"Found min_cost theta1: {self.theta1} theta2:{self.theta2} "
              f"s1: {self.sim_model.model.im_min_s} s2: {self.sim_model.model.im_max_s}"
              f"avg cost: {self.exp_res_avg_cost}"
              f"avg otd: {self.exp_res_avg_otd}"
              f"avg fulfill demand: {self.exp_res_avg_fulfill_demand}")

    def get_cost_delta(self):
        self.fit_hist_cost_ols()
        # print(self.fit_ols_res.summary())
        if self.fit_cost_ols_res.rsquared >= 0.6:
            self.cost_delta.theta1 = -10 * self.fit_cost_ols_res.params[1]/abs(self.fit_cost_ols_res.params[2])
            self.cost_delta.theta2 = -10 * self.fit_cost_ols_res.params[2]/abs(self.fit_cost_ols_res.params[1])
        else:
            if self.exp_res_avg_cost_prev > self.exp_res_avg_cost:
                self.cost_delta.theta1 = -15
                self.cost_delta.theta2 = -10
            else:
                self.cost_delta.theta1 = -5
                self.cost_delta.theta2 = -5

    def fit_hist_cost_ols(self):
        X = np.column_stack(self.hist_exog_vars)
        X = np.column_stack(X)
        y = np.array(self.hist_endog_cost)
        rsm = RSModel(X, y)
        rsm.fit_ols()
        self.fit_cost_ols_res = rsm.fit_ols_res
        self.rsm = rsm

    def block_run(self, block_run, theta1,  theta2):
        run_num = 1
        while run_num <= block_run:
            self.reset_im(theta1, theta2)
            self.one_exp_data()
            run_num += 1

    def reset_im(self, theta1, theta2):
        self.theta1 = theta1
        self.theta2 = theta2
        self.reset_imp_bound()
        new_min_s = self.sim_model.model.min_s_lower_bound + self.theta1
        new_max_s = new_min_s + self.theta2
        new_imp = {"im_min_s": new_min_s, "im_max_s": new_max_s}
        self.sim_model.model.reset_im_policy(new_imp)

    def reset_imp_bound(self):
        """add some salt"""
        self.sim_model.model.min_s_lower_bound = 100
        self.sim_model.model.max_s_upper_bound = 5000

    def one_exp_data(self):
        block_run = BlockRun(self.block_ctl_rand)
        block_run.block_avg(self.sim_model, True)
        self.exp_endog_cost.append(block_run.avg_cost)
        self.exp_endog_otd.append(block_run.avg_otd)
        self.hist_endog_cost.append(block_run.avg_cost)
        self.hist_endog_otd.append(block_run.avg_otd)
        self.hist_endog_fulfill.append(block_run.avg_fulfill_demand)
        self.exp_endog_fulfill.append(block_run.avg_fulfill_demand)
        self.exp_exog_vars.append([1., self.theta1,self.theta2])
        self.hist_exog_vars.append([1., self.theta1,self.theta2])


def plot_reg_model(fit_ols_res):
    pred_ols = fit_ols_res.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    pred_ols_mean = pred_ols.summary_frame()["mean"]

    i = 1
    while i < len(fit_ols_res.model.exog[0]):
        fig, ax = plt.subplots(figsize=(8, 6))
        x = fit_ols_res.model.exog[:, i]
        y = fit_ols_res.model.endog
        ax.plot(x, y, "o", label="Data")
        ax.plot(x, y, "b-", label="True")
        ax.plot(x, pred_ols_mean, "r--.", label="Predicted")
        ax.plot(x, iv_u, "r--")
        ax.plot(x, iv_l, "r--")
        i += 1


class RSModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fit_ols(self):
        self.fit_ols_mdl = lm.OLS(self.y, self.X)
        self.fit_ols_res = self.fit_ols_mdl.fit()


class RepeatRun:

    exp2f2l = Exp2F2L()
    exprec = []
    exp_count = []

    def run(self, counts):
        i = 0
        while i <= counts:
            self.exp2f2l.block_ctl_rand = crn(30)
            print(f"Experiment Run: {i} use CRN number {self.exp2f2l.block_ctl_rand.seed_num }")
            if (i % 2) == 0:
                theta1 = 100
                theta2 = 600
            else:
                theta1 = self.exp2f2l.theta1
                theta2 = self.exp2f2l.theta2
            self.exp2f2l.find_feasible_center(block_run=5, init_theta1=theta1,
                                              init_theta2=theta2, otd_threshold=0.90)
            self.exp2f2l.find_min_cost(block_run=5, theta1=self.exp2f2l.theta1, theta2=self.exp2f2l.theta2,
                                       otd_threshold=0.90)
            self.exp_count.append(i)
            linerec = self.linerec(i)
            self.exprec.append(linerec)
            i += 1

    def linerec(self, count):
        exprec = ExpRec
        c0:exprec.exp = count
        c1:exprec.theta1 = self.exp2f2l.theta1
        c2:exprec.theta2 = self.exp2f2l.theta2
        c3:exprec.min_s = self.exp2f2l.sim_model.model.im_min_s
        c4:exprec.max_s = self.exp2f2l.sim_model.model.im_max_s
        c5:exprec.avg_cost = self.exp2f2l.exp_res_avg_cost
        c6:exprec.avg_otd = self.exp2f2l.exp_res_avg_otd
        c7:exprec.avg_fulfill = self.exp2f2l.exp_res_avg_fulfill_demand
        c8:exprec.avg_period_cost = self.exp2f2l.exp_res_avg_cost / self.exp2f2l.sim_model.model.periods
        c9:exprec.avg_unit_cost = self.exp2f2l.exp_res_avg_cost / self.exp2f2l.exp_res_avg_fulfill_demand

        return [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]

    def asdf(self):
        self.df = pd.DataFrame(np.array(self.exprec), columns=["exp","theta1","theta2","min_s","max_s","avg_cost",
                                                               "avg_otd","avg_fulfill","avg_period_cost","avg_unit_cost"])

    def fit_ols(self):
        self.asdf()
        y, X = pt.dmatrices('avg_unit_cost~min_s + max_s', data=self.df, return_type='dataframe')
        self.fit_ols_mdl = lm.OLS(y, X)
        self.fit_ols_res = self.fit_ols_mdl.fit()

    def fit_quard_ols(self):
        self.asdf()
        y, X = pt.dmatrices('avg_unit_cost ~ min_s + min_s**2 + max_s + max_s**2 + min_s*max_s',
                            data=self.df, return_type='dataframe')
        self.fit_ols_mdl = lm.OLS(y, X)
        self.fit_ols_res = self.fit_ols_mdl.fit()

    def plot_trend(self):
        sns.set_theme(style="darkgrid")
        self.asdf()
        # Plot the responses for different events and regions
        sns.lineplot(x="exp", y="avg_period_cost",
                     data=self.df)


if __name__ == "__main__":
    exp2f2l = RepeatRun()
    exp2f2l.run(100)
    exp2f2l.fit_ols()
    plot_reg_model(exp2f2l.fit_ols_res)
    exp2f2l.plot_trend()
    print(f"{exp2f2l.df}")

    breakpoint()
