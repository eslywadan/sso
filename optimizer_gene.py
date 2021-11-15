from simulation import Simulation, IMPOptMea
from matplotlib import pyplot as plt
import numpy as np


class Chromosome:
    def __init__(self,theta1, theta2):
        self.theta_min_s = theta1
        self.theta_max_s = theta2


class Chromosomes:

    def __init__(self,cost=None, theta_min_s=0, theta_max_s=0) :
        self.cost = cost
        self.chromosome = Chromosome(theta_min_s, theta_max_s)

    def crossover(self):
        pass

    def mutation(self):
        pass


class Population:

    def __init__(self, cr=10, cc=10, cm=10):
        self.pop_random_max_num = cr
        self.pop_crossover_max_num = cc
        self.pop_mutation_max_num = cm
        self.random = set()
        self.cross_over = set()
        self.mutation = set()

    def add_random(self):
        while len(self.random) < self.pop_random_max_num:
            theta_min_s = np.random.uniform(0, 1000, size=1)
            theta_max_s = np.random.uniform(0, 2000, size=1)
            self.random.add(Chromosome(theta_min_s, theta_max_s))

    def add_crossover(self):
        self.cross_over.add(Chromosomes(self))


class Generation:
    gens: int
    pop: Population
    best_pop: Population

    def __init__(self,generation):
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
        self.chromosome_all_gen_pools = []
        self.chromosome_cur_pop_pools = []

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
                new_min_s = self.model.min_s_lower_bound + self.gen_best_chromosome.chromosome.theta_min_s
                new_max_s = new_min_s + self.gen_best_chromosome.chromosome.theta_max_s
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
                # self.cur_pop_best_chromosome = {f"cost:{self.cur_pop_best}, theta1:{self.theta_min_s}, theta2:{self.theta_max_s}"}
                self.cur_pop_best_chromosome = Chromosomes(self.cur_pop_best, self.theta_min_s, self.theta_max_s)
                print(f"S2:New Theta has committed the Cumulative OTD rate and lower cost, sol accept~")
                self.pop_count += 1
                continue

        if self.cur_pop_best and self.cur_pop_best_chromosome is not None:
            self.add_candidate("cur_pop", self.cur_pop_best_chromosome)
            self.gen_res = "S2"

    def pop_candidate(self):
        self.theta_min_s = np.random.uniform(0, 1000, size=1)
        self.theta_max_s = np.random.uniform(0, 2000, size=1)

    def add_candidate(self, base_line, chromosome):
        sol = chromosome
        if base_line == "all_gen":
            self.chromosome_all_gen_pools.append(sol)
        if base_line == "cur_pop":
            self.chromosome_cur_pop_pools.append(sol)




