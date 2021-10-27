import statistics as sta
import numpy as np


class InvMdlSingle:
    """Inventory Model for a single product"""
    def __init__(self, env_set, cost_set, stochastic_set, min_s_max_s_set):
        # env_set
        self.min_cost_im_min_s = None
        self.min_cost_im_max_s = None
        self.periods = env_set["periods"]
        self.start_inv = [0] * self.periods
        self.init_inv = env_set["init_inv"]
        self.start_inv[0] = env_set["init_inv"]
        self.back_order = [0] * self.periods
        self.init_backorder = env_set["init_back_order"]
        self.back_order[0] = env_set["init_back_order"]
        # cost_set
        self.unit_holding_cost = cost_set["holding_cost"]
        self.fix_PO_cost = cost_set["fix_PO_cost"]
        self.unit_po_cost = cost_set["Unit_PO_Cost"]
        # stochastic_set
        self.demand_avg = stochastic_set["demand_avg"]
        self.order_lt_mu = stochastic_set["lt_mu"]
        # IM Policy Set min_s_max_s_set
        self.im_min_s = min_s_max_s_set["im_min_s"]
        self.im_max_s = min_s_max_s_set["im_max_s"]

    def __reset_demand(self):            # Reset Demand
        self.demand = [] * self.periods

    def __reset_order_lt(self):          # Reset order lead time
        self.order_lt = [] * self.periods

    def reset_plan(self):              # Reset the plan
        # Reset Plan Qty
        self.order_received = [0] * self.periods
        self.fulfill_qty = [0] * self.periods
        self.end_inv = [0] * self.periods
        self.open_order = [0] * self.periods
        self.order_placed = [0] * self.periods
        self.inv_position = [0] * self.periods
        # Reset Cumulative OTD
        self.cum_otd = 1.0
        self.cum_demand_qty = 0
        self.cum_fulfill_qty = 0
        self.cum_nsfs_qty = 0
        # Rest Cost
        self.cum_cost = 0
        self.period_cost = [0] * self.periods
        self.holding_cost = [0] * self.periods
        self.order_cost = [0] * self.periods
        self.cum_po_cost = 0
        self.cum_holding_cost = 0

    def reset_im_policy(self, min_s_max_s_set):
        if min_s_max_s_set["im_min_s"] < self.min_s_lower_bound:
            min_s_max_s_set["im_min_s"] = self.min_s_lower_bound

        if min_s_max_s_set["im_max_s"] > self.max_s_upper_bound:
            min_s_max_s_set["im_max_s"] = self.max_s_upper_bound

        self.im_min_s = min_s_max_s_set["im_min_s"]
        self.im_max_s = min_s_max_s_set["im_max_s"]

    def init_ss(self, sample_count=1000, use_the_last_best=False):
        """init_ss will estimate the coverage by considering  average demand and standard lead time"""
        sim_demand = GenDemand.gen_demand(self.demand_avg, sample_count)
        sim_lt = GenLeadTime.gen_lead(self.order_lt_mu, sample_count)
        demand_quantile = sta.quantiles(sim_demand, n=10)
        lt_quantile = sta.quantiles(sim_lt, n=10)
        self.im_min_s = round(demand_quantile[6] * lt_quantile[6])
        self.im_max_s = round(demand_quantile[8] * lt_quantile[8])
        self.min_s_lower_bound = round(demand_quantile[1] * lt_quantile[1])
        self.max_s_upper_bound = round(demand_quantile[8] * lt_quantile[8])
        if use_the_last_best and self.min_cost_im_min_s is not None:
            self.im_min_s = self.min_cost_im_min_s
            self.im_max_s = self.min_cost_im_max_s

    def gen_demand(self):
        self.__reset_demand()
        self.demand = list(GenDemand.gen_demand(self.demand_avg, self.periods).round())
        # init plan kpi
        self.nsfs_qty = [0] * self.periods

    def gen_lt(self):
        self.__reset_order_lt()
        self.order_lt = list(GenLeadTime.gen_lead(self.order_lt_mu, self.periods).round())

    def __demand_fulfill(self, period):
        available = self.start_inv[period] + self.order_received[period]
        requirement = self.demand[period] + self.back_order[period]
        backorder = requirement - available

        if backorder <= 0:  # Demand is fully fulfilled
            fulfill_qty = self.demand[period]
            end_inv = available-fulfill_qty
            backorder = 0
        else:      # If backorder > 0 , exist un-fulfill demand
            fulfill_qty = available
            end_inv = 0

        demand_fulfill = {
            "Period": period,
            "Demand": self.demand[period],
            "starting_inv": self.start_inv[period],
            "order_received": self.order_received[period],
            "fulfill_qty": fulfill_qty,
            "end_inv": end_inv,
            "back_order": backorder
        }
        return demand_fulfill

    def __bal_by_period(self, period):
        """bal by period will cal
                   ending_inv( back_order ) = demand - (starting_inv+order_received)"""
        fulfill_res = self.__demand_fulfill(period)
        self.fulfill_qty[period] = fulfill_res['fulfill_qty']
        self.end_inv[period] = fulfill_res['end_inv']
        self.nsfs_qty[period] = fulfill_res['back_order']   # nsfs : not supply from stock on-hand
        self.cum_demand_qty += fulfill_res['Demand']
        self.cum_fulfill_qty += fulfill_res['fulfill_qty']
        self.cum_nsfs_qty += fulfill_res['back_order']
        self.inv_position[period] = self.end_inv[period] + self.open_order[period]

        if self.inv_position[period] < self.im_min_s:  # Decision point on re-order
            order_qty = self.im_max_s - self.inv_position[period]
            self.order_placed[period] = order_qty
            receipt_period = self.order_lt[period]+period+1
            self.__add_open_order(period + 1, receipt_period - 1, order_qty)
            if receipt_period < self.periods:
                self.order_received[receipt_period] += order_qty

        next_period = period + 1      # Let the next period's starting inv be this period's ending env
        if next_period < self.periods:
            self.start_inv[next_period] = self.end_inv[period]
            self.back_order[next_period] = fulfill_res['back_order']

        self.__update_cost(period)      # Update the cost due to this period

    def __update_cost(self, period):
        order_cost = 0
        order_placed = self.order_placed[period]
        if order_placed > 0:
            self.order_cost[period] = order_placed * self.unit_po_cost + self.fix_PO_cost
        self.holding_cost[period] = self.end_inv[period] * self.unit_holding_cost
        self.period_cost[period] = self.holding_cost[period] + self.order_cost[period]
        self.cum_cost += self.period_cost[period]
        self.cum_po_cost += self.order_cost[period]
        self.cum_holding_cost += self.holding_cost[period]

    def __add_open_order(self, p_start, p_end, order_qty):
        period = p_start
        if p_end >= self.periods-1:
            p_end = self.periods-1
        while period <= p_end:
            self.open_order[period] += order_qty
            period += 1

    def __cal_cum_otd(self):
        cum_require_qty = sum(self.demand)
        cum_fulfill_qty = self.cum_fulfill_qty
        on_time_deliver_rate = cum_fulfill_qty/cum_require_qty
        return on_time_deliver_rate

    def bal_all(self):
        period = 0
        while period < self.periods:
            self.__bal_by_period(period)
            period += 1

        self.cum_otd = self.__cal_cum_otd()

    def show_by_period(self, period):
        """The function will show the balance result for given period"""
        print(f" Period : {period} Stock Policy (S,s): ({self.im_max_s},{self.im_min_s})\n"
              f" Starting Inv : {self.start_inv[period]} \n"
              f" Demand : {self.demand[period]}\n"
              f" Order Received: {self.order_received[period]}\n"
              f" Ending Inventory: {self.end_inv[period]}\n"
              f" Back Order: {self.back_order[period]}\n"
              f" Fulfill qty: {self.fulfill_qty[period]}\n"
              f" Open Order Qty : {self.open_order[period]}\n"
              f" Inventory Position: {self.inv_position[period]}\n"
              f" Order Placed: {self.order_placed[period]}\n"
              f" Order LT: {self.order_lt[period]}\n"
              f" Period Cost: {self.period_cost[period]}\n "
              f" Cum OTD : {self.cum_otd}\n")


class GenDemand:
    """Generate demand"""
    @staticmethod
    def gen_demand(mu: int = 100, size: int = 100):
        dem = np.random.exponential(mu, size)
        return dem


class GenLeadTime:
    """Generate Lead Time"""
    @staticmethod
    def gen_lead(theta=6, size: int = 100):
        lead = np.random.poisson(theta, size)
        return lead



