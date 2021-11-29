import numpy as np


class ControlRandomNumber:
    """ControlRandomNumber is used for controlling a set of seed elements forming a controlled random env"""

    def __init__(self,seed_num):
        """seed_seq is a list with seed elelmet"""
        self.seed_num = seed_num
        self.seed_seq = []
        self.gen_seed()
        self.cur_seed_ix = 0

    def gen_seed(self):
        while len(self.seed_seq) < self.seed_num:
            self.seed_seq.append(np.random.randint(2^32))

        self.cur_seed = self.seed_seq[0]

    def next_as_cur_ix(self):
        self.cur_seed_ix += 1
        if self.cur_seed_ix > len(self.seed_seq)-1:
            self.cur_seed_ix = 0

        self.cur_seed = self.seed_seq[self.cur_seed_ix]


if __name__ == '__main__':
    t_crn = ControlRandomNumber(3)
    print(t_crn.cur_seed)
    t_crn.next_as_cur_ix()
    print(t_crn.cur_seed)
    t_crn.next_as_cur_ix()
    print(t_crn.cur_seed)
    t_crn.next_as_cur_ix()
    print(t_crn.cur_seed)

    breakpoint()
