from time import time


def timer(func):
    """timer is a decorator method used for returning the elapsed time during
    executing a function
    https://lotabout.me/2017/Python-Decorator/
    """

    def wrapper(x, y=10):
        before = time()
        result = func(x, y)
        after = time()
        print(f"elapsed time {after - before}")
        return result

    return wrapper


def add(x, y=10):
    return x + y


add = timer(add)


def sub(x, y=10):
    return x - y


sub = timer(sub)


def dummy():
    pass


print("add(10", add(10))
print("add(20, 30)", add(20, 30))
dummy()


class


