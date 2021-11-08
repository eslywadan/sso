from simulation import Simulation, IMPOptMea
from matplotlib import pyplot as plt
import numpy as np


class Chromosome:
    def __init__(self, generation):
        self.gen = generation
        self.theta_min_s = 0
        self.theta_max_s = 0


class Population:
    chromosome: Chromosome

    def __init__(self, generation):
        self.chromosome = Chromosome(generation)


class Generation:
    gens: int
    pop: Population
    best_pop: Population

    def __init__(self,generation):
        self.pop = Population(generation)
        self.gens = generation


class OptimizeTotalCostGene(Simulation, Generation):

    imp_opt_mea = IMPOptMea
    generation: Generation
    population: Population
    chromosome: Chromosome
    print_res = None

    def __init__(self, p1, p2, p3, p4):
        Simulation.__init__(self, p1, p2, p3, p4)
        self.regen_hist = []
        self.sim_init()

    def gene(self,max_measure=100):
        """The function will keep the best solution for accumulated generations"""
        self.gene_seq = []
        self.top_measure = 0
        self.sim_measure = 0
        self.min_cost_measure = self.top_measure
        self.min_cost = None
        self.min_cost_im_min_s = None
        self.min_cost_im_max_s = None
        self.gene_dim = 2
        self.cotd_threshold = 0.9
        self.gene_latest_best = 0.0
        self.gene_latest_best_measure = 0
        self.opt_hist_tc = []
        self.opt_hist_min_s = []
        self.opt_hist_max_s = []

        self.gene = 0
        self.gen_best_chromosome = None

        while self.top_measure <= max_measure:   # max_measure restrict the max measure of generation
            self.opt_ss_gene()
            if self.top_measure == 1:
                self.init_cost = self.model.cum_cost
                self.min_cost = self.init_cost
                self.min_cost_measure = self.top_measure

            if self.min_cost is None:
                self.min_cost = self.model.cum_cost

            if self.gen_res == 'S2' and self.cur_pop_best < self.min_cost:
                self.min_cost = self.cur_pop_best
                self.gene_latest_best = self.min_cost
                self.gene_latest_best_measure = self.top_measure
                self.gen_best_chromosome = self.cur_pop_best_chromosome
                self.min_cost_measure = self.top_measure
                new_min_s = self.model.min_s_lower_bound + self.gen_best_chromosome["theta1"]
                new_max_s = new_min_s + self.gen_best_chromosome["theta2"]
                self.min_cost_im_min_s = new_min_s
                self.min_cost_im_max_s = new_max_s

            self.opt_hist_tc.append(self.model.cum_cost)
            self.opt_hist_min_s.append(self.model.im_min_s)
            self.opt_hist_max_s.append(self.model.im_max_s)
            self.top_measure += 1
            self.gene += 1

        opt_plot = plt
        opt_plot.plot(self.opt_hist_tc)

    def opt_ss_gene(self,pop_num=10):

        """ the function will keep the best solution for this generation"""
        self.pop_count = 0
        self.cur_pop_best = None
        self.cur_pop_best_chromosome = None
        self.gen_res = None

        while self.pop_count < pop_num:
            self.pop_candidate()
            new_theta1 = self.theta_min_s
            new_theta2 = self.theta_max_s

            if new_theta1 < 0:
                new_theta1 = 0
            if new_theta2 < 0:
                new_theta2 = 0

            new_min_s = self.model.min_s_lower_bound + new_theta1
            new_max_s = new_min_s + new_theta2
            new_imp = {"im_min_s": new_min_s, "im_max_s": new_max_s}
            self.model.reset_im_policy(new_imp)
            yk_new = self.sim_sc1("y(Theta_k+1")

            if yk_new["cum_otd"] < self.cotd_threshold:
                print(f"F1:New Theta has the Cumulative OTD rate less than threshold, sol fail~")
                self.pop_count += 1
                continue

            if self.cur_pop_best is None:
                self.cur_pop_best = yk_new["cum_cost"]
            if yk_new["cum_cost"] < self.cur_pop_best:
                self.cur_pop_best = yk_new["cum_cost"]
                self.cur_pop_best_chromosome = {"theta1":self.theta_min_s, "theta2": self.theta_max_s}
                print(f"S2:New Theta has committed the Cumulative OTD rate and lower cost, sol accept~")
                self.pop_count += 1
                continue

        if self.cur_pop_best:
            self.add_candidate(self.cur_pop_best_chromosome)
            self.gen_res = "S2"

    def pop_candidate(self):
        self.theta_min_s = np.random.uniform(0, 1000, size=1)
        self.theta_max_s = np.random.uniform(0, 2000, size=1)

    def add_candidate(self, candidate: Chromosome):
        pass


