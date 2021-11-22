from simulation import Scatter2Dims
from optimizer_fdsa import OptimizeTotalCostFDSA
from optimizer_spsa import OptimizeTotalCostSPSA
from optmizer_gene00 import OptimizeTotalCostGene


class PlanEnv:
    # planning env setting static
    env_set = {"periods": 50, "init_inv": 1000, "init_back_order": 0}
    # Cost Setting
    cost_set = {"holding_cost": 1, "fix_PO_cost": 36, "Unit_PO_Cost": 2}
    # Stochastic Setting
    stochastic_set = {"demand_avg": 100, "lt_mu": 6}
    # s, S stock policy set
    min_s_max_s_set = {"im_min_s": 600, "im_max_s": 10000}


def singleprodfixdemandfdsa(regen_count=10):
    plan_env = PlanEnv
    opt = OptimizeTotalCostFDSA(p1=plan_env.env_set, p2=plan_env.cost_set, p3=plan_env.stochastic_set,
                                p4=plan_env.min_s_max_s_set)
    opt.regen_count = 0
    while opt.regen_count < regen_count:
        opt.fdsa(ak=1, max_measure=300, use_the_last_best=False)
        opt.regen_count += 1
        opt.regen_hist.append(
            {"Regen Count": opt.regen_count, "Min Cost Measure": opt.min_cost_measure, "Min Cost": opt.min_cost})
        print(f"regen count: {regen_count}\n"
              f"min cost: {opt.min_cost}\n"
              f"min cost measure: {opt.min_cost_measure}\n"
              f"min cost min s {opt.min_cost_im_min_s}\n"
              f"min cost max s {opt.min_cost_im_max_s}\n")
        
        Scatter2Dims.plot_dist_scatter(opt.opt_hist_min_s, opt.opt_hist_max_s)


def dummy():
    print("break")


# Press the green button in the gutter to run the script.
def singleprodfixdemandspsa(regen_count=10):
    plan_env = PlanEnv
    opt2 = OptimizeTotalCostSPSA(p1=plan_env.env_set, p2=plan_env.cost_set, p3=plan_env.stochastic_set,
                                p4=plan_env.min_s_max_s_set)
    opt2.regen_count = 0
    while opt2.regen_count < regen_count:
        opt2.spsa(max_measure=300)
        opt2.regen_count += 1
        opt2.regen_hist.append(
            {"Regen Count": opt2.regen_count, "Min Cost Measure": opt2.min_cost_measure, "Min Cost": opt2.min_cost})
        print(f"regen count: {regen_count}\n"
              f"min cost: {opt2.min_cost}\n"
              f"min cost measure: {opt2.min_cost_measure}\n"
              f"min cost min s {opt2.min_cost_im_min_s}\n"
              f"min cost max s {opt2.min_cost_im_max_s}\n")
    dummy()


def singleprodfixdemandgene(regen_count=10):
    plan_env = PlanEnv
    opt3 = OptimizeTotalCostGene(p1=plan_env.env_set, p2=plan_env.cost_set, p3=plan_env.stochastic_set,
                                p4=plan_env.min_s_max_s_set)
    opt3.regen_count = 0
    while opt3.regen_count < regen_count:
        opt3.gene(max_measure=300)
        opt3.regen_count += 1
        opt3.regen_hist.append(
            {"Regen Count": opt3.regen_count, "Min Cost Measure": opt3.min_cost_measure, "Min Cost": opt3.min_cost})
        print(f"regen count: {regen_count}\n"
              f"min cost: {opt3.min_cost}\n"
              f"min cost measure: {opt3.min_cost_measure}\n"
              f"min cost min s {opt3.min_cost_im_min_s}\n"
              f"min cost max s {opt3.min_cost_im_max_s}\n")
    dummy()

if __name__ == '__main__':
    # Scatter2Dims.plot_stochastic_demand_lt_scatter()
    # Scatter2Dims.plot_normal_scatter()
    # dummy()
    # Segmenting initial setting into 4 types

    # singleprodfixdemandfdsa(regen_count=5)

    # singleprodfixdemandspsa(regen_count=1)

    singleprodfixdemandgene(regen_count=1)

    dummy()
