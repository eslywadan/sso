import numpy as np
"""Reference: https://towardsdatascience.com/a-simple-guide-to-linear-regression-using-python-7050e8c751c1 """
import statsmodels.regression.linear_model as lm
import statsmodels.api as sm
from controlled_random import ControlRandomNumber as crn
from simulation import Simulation
from main import PlanEnv as plan_env


class Experiment:
    pass


class OtdDelta:
    theta1: float
    theta2: float

    def __init__(self):
        self.theta1 = 0
        self.theta2 = 0


class Exp2F2L(Experiment):
    theta1:float
    theta2:float

    def __init__(self):
        self.reset_exp_res()
        self.otd_delta = OtdDelta()
        self.sim_model = Simulation(p1=plan_env.env_set, p2=plan_env.cost_set, p3=plan_env.stochastic_set,
                                    p4=plan_env.min_s_max_s_set)
        self.block_ctl_rand = crn(10)
        self.hist_exog_vars = []
        self.hist_endog_cost = []
        self.hist_endog_otd = []

    def reset_exp_res(self):
        self.exp_exog_vars = []
        self.exp_endog_cost = []
        self.exp_endog_otd = []

    def find_feasible_center(self, block_run=5, init_theta1=100, init_theta2=600, otd_threshold=0.80):
        self.block_run(block_run, init_theta1, init_theta2)
        self.exp_res_avg_otd = sum(self.exp_endog_otd)/len(self.exp_endog_otd)
        self.otd_threshold = otd_threshold

        while self.exp_res_avg_otd < otd_threshold:
            self.exp_res_avg_otd_prev = self.exp_res_avg_otd
            self.check_otd_delta()
            theta1 = self.theta1 + self.otd_delta.theta1
            theta2 = self.theta2 + self.otd_delta.theta2
            self.reset_exp_res()
            self.block_run(block_run,theta1,theta2)
            self.exp_res_avg_otd = sum(self.exp_endog_otd) / len(self.exp_endog_otd)
            print(f"average otd: {self.exp_res_avg_otd}")

        print(f"Found feasible theta1: {self.theta1} theta2:{self.theta2}")

    def check_otd_delta(self):
        self.fit_ols()
        print(self.fit_ols_res.summary())
        if self.exp_res_avg_otd_prev < self.exp_res_avg_otd:
            self.otd_delta.theta1 = 50
            self.otd_delta.theta2 = 40
        else:
            self.otd_delta.theta1 = 10
            self.otd_delta.theta2 = 20

    def fit_ols(self):
        X = np.column_stack(self.hist_exog_vars)
        X = np.column_stack(X)
        y = np.array(self.hist_endog_otd)
        rsm = RSModel(X, y)
        rsm.fit_ols()
        self.fit_ols_res = rsm.fit_ols_res

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
        self.exp_exog_vars.append([1., self.theta1,self.theta2])
        self.hist_exog_vars.append([1., self.theta1,self.theta2])


class BlockRun:

    def __init__(self, ctl_ran:crn):
        self.ctl_ran = ctl_ran
        self.avg_cost = 0
        self.sum_cost = 0
        self.avg_otd = 0
        self.sum_otd = 0

    def block_avg(self, sim_model:Simulation, shuffle=True):
        if shuffle is True:
            np.random.shuffle(self.ctl_ran.seed_seq)
            print(f"Shuffle is {shuffle}, the random seed is shuffled!")
        while self.ctl_ran.cur_seed_ix < self.ctl_ran.seed_num-1:
            sim_model.sim_simple(self.ctl_ran.cur_seed)
            self.sum_cost += sim_model.model.cum_cost
            self.sum_otd += sim_model.model.cum_otd
            print(f"min_s:{sim_model.model.im_min_s},max_s:{sim_model.model.im_max_s}")
            self.ctl_ran.next_as_cur_ix()

        self.ctl_ran.next_as_cur_ix()
        self.avg_cost = self.sum_cost/self.ctl_ran.seed_num
        self.avg_otd = self.sum_otd/self.ctl_ran.seed_num


class RSModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fit_ols(self):
        self.fit_ols_mdl = lm.OLS(self.y, self.X)
        self.fit_ols_res = self.fit_ols_mdl.fit()


if __name__ == "__main__":
    exp2f2l = Exp2F2L()
    exp2f2l.find_feasible_center(otd_threshold=0.85)


    breakpoint()