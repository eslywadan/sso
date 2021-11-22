from optimizer_gene import Chromosome, Chromosomes, Population, Generation, Genrepo, OptToCostGene
from main import PlanEnv


def dummy():
    print("break")


plan_env = PlanEnv
opt1 = OptToCostGene(p1=plan_env.env_set, p2=plan_env.cost_set, p3=plan_env.stochastic_set, p4=plan_env.min_s_max_s_set)
opt1.sim_measure = 0

opt1.gens(1000)
opt1.sim_gens()
sol = sorted(opt1.generation.gensol, key = lambda s: s[1])

dummy


