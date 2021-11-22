#
from timeit import default_timer as timer


class TimingDecorator(object):
    def __init__(self, klas):
        self.klas = klas
        self.org_calculation1 = klas.calculation1
        self.klas.calculation1 = self.calculation1

    def __call__(self, arg=None):
        return self.klas.__call__()

    def calculation1(self, *args, **kwargs):
        start = timer()
        print(self.org_calculation1(self.klas, *args, **kwargs))
        end = timer()
        print(end - start)


@TimingDecorator
class MyFancyClassTest():

    def calculation1(self, a, b):
        from time import sleep
        # i'm adding a sleep - as a test case if counter works.
        sleep(2)
        print(f"a:{a+b*2}, b:{3*a}")
        self.ret = (a+b*2)/(3*a)

fancy_object = MyFancyClassTest()
fancy_object.calculation1(50, 215)