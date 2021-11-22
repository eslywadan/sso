from simulation import Simulation, IMPOptMea
import numpy as np
import random


class Chromosome:
    def __init__(self, gentype=1):
        self.gentype = gentype
        if self.gentype == 1:
            self.random()
        if self.gentype == 2:
            self.crossover()
        if self.gentype == 3:
            self.mutation()

    def random(self):
        self.theta_min_s = np.random.uniform(0, 1000, size=1)
        self.theta_max_s = np.random.uniform(0, 2000, size=1)

    def crossover(self, genrepo):
        max_ix = len(genrepo.gensol)-1
        if max_ix <= 0:
            self.random()
            return
        xc = random.randint(0, max_ix)
        yc = random.randint(0, max_ix)
        new_x = genrepo.gensol[xc]
        new_y = genrepo.gensol[yc]
        self.theta_min_s = new_x[0].theta_min_s
        self.theta_max_s = new_y[0].theta_max_s

    def mutation(self, genrepo):
        max_ix = len(genrepo.gensol)-1
        if max_ix <= 0:
            self.random()
            return
        xc = random.randint(0, max_ix)
        new_ch = genrepo.gensol[xc]
        si = random.randint(0,1)
        if si == 0:
            self.theta_min_s = new_ch[0].theta_min_s
            self.theta_max_s = np.random.uniform(0, 2000, size=1)
        if si == 1:
            self.theta_min_s = np.random.uniform(0, 1000, size=1)
            self.theta_max_s = new_ch[0].theta_max_s


class Chromosomes:

    def __init__(self,chromo: Chromosome, cost=None) :
        self.cost = cost
        self.chromosome = chromo


class Population:

    def __init__(self, generation, cr=10, cc=10, cm=10):
        self.generation = generation
        self.pop_random_max_num = cr
        self.pop_crossover_max_num = cc
        self.pop_mutation_max_num = cm
        self.random = []
        self.cross_over = []
        self.mutation = []

    def add_random(self, size=None):
        if size is None:
            size = self.pop_random_max_num
        while len(self.random) < size:
            chromo = Chromosome(1)
            random_chromo = Chromosomes(chromo, cost=None)
            self.random.append(random_chromo)

    def add_crossover(self, genrepo, size=None):
        if size is None:
            size = self.pop_crossover_max_num
        while len(self.cross_over) < size:
            chromo = Chromosome()
            cross_over_chromo = Chromosomes(chromo.crossover(genrepo), cost=None)
            self.cross_over.append(cross_over_chromo)

    def add_mutation(self,genrepo, size=None):
        if size is None:
            size = self.pop_mutation_max_num
        while len(self.mutation) < size:
            chromo = Chromosome()
            mutate_chromo = Chromosomes(chromo.mutation(genrepo), cost=None)
            self.mutation.append( mutate_chromo )


class Generation:
    gens: int
    pop: Population
    best_pop: Population

    def __init__(self,generation):
        self.gens = generation
        self.pop = Population(generation, cr=10, cc=10, cm=10)
        self.gensol = []


class Genrepo(Generation):
    """Genrepo is treated as a particular generation"""

    def __init__(self, max_pool):
        self.gens = 0
        self.max_pool = max_pool
        self.gensol = []

    def add_sol(self, sols):
        before_num = len(self.gensol)
        for sol in sols:
            self.gensol.append(sol)

        after_num = len(self.gensol)
        if after_num <= before_num:
            print(f"Original gensol {before_num}")
            print(f"After gensol {after_num}")
            breakpoint()

        self.prun_sol()

    def prun_sol(self):
        sols = sorted(self.gensol, key=lambda s: s[1])
        self.gensol = sols[0:self.max_pool-1]


class OptToCostGene(Simulation, Generation):

    imp_opt_mea = IMPOptMea
    generation: Generation
    population: Population
    chromosome: Chromosome
    print_res = None

    def __init__(self, p1, p2, p3, p4):
        Simulation.__init__(self, p1, p2, p3, p4)
        self.regen_hist = []
        self.sim_init()
        self.cotd_threshold = 0.9

    def init_genrepo(self, max_iter, max_pool):
        self.generation = Generation(0)
        self.generation.pop.add_random(max_iter)
        self.sim_gens(self.generation.pop.random)
        self.genrepo = Genrepo(max_pool)
        self.genrepo.add_sol(self.generation.gensol)

    def loop_gens(self, gensiter, cr, cc, cm):
        for gens in gensiter:

            self.generation = Generation(gens)
            self.generation.gensol = []
            self.generation.pop.add_random(cr)
            self.sim_gens(self.generation.pop.random)
            self.generation.pop.add_mutation(self.genrepo, cm)
            self.sim_gens(self.generation.pop.mutation)
            self.generation.pop.add_crossover(self.genrepo, cc)
            self.sim_gens(self.generation.pop.cross_over)
            self.genrepo.add_sol(self.generation.gensol)
            print(f"Generation: {self.generation} Gens Sol {len(self.generation.gensol)}")
            print(f"GenRepo {len(self.genrepo.gensol)}")

    def sim_ss_gene(self, chromo):
        new_min_s = self.model.min_s_lower_bound + chromo.theta_min_s
        new_max_s = new_min_s + chromo.theta_max_s
        new_imp = {"im_min_s": self.model.min_s_lower_bound + chromo.theta_min_s, "im_max_s": new_max_s}
        self.model.reset_im_policy(new_imp)
        self.sim_sc1("sim ss gen")

    def sim_gens(self, chromosomes):
        for gen in chromosomes:
            if gen.chromosome is None:
                continue
            self.sim_ss_gene(gen.chromosome)
            if self.model.cum_otd < self.cotd_threshold:
                continue
            self.generation.gensol.append((gen.chromosome,self.model.cum_cost))




if __name__ == '__main__':
    from main import PlanEnv
    plan_env = PlanEnv
    opt1 = OptToCostGene(p1=plan_env.env_set, p2=plan_env.cost_set, p3=plan_env.stochastic_set, p4=plan_env.min_s_max_s_set)
    opt1.sim_measure = 0

    opt1.init_genrepo(1000, 40)
    gensiter = list(range(1, 10))
    opt1.loop_gens(gensiter, 10, 10, 10)
    print(opt1.sim_measure)
    print("Good!")
