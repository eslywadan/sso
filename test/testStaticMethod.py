class Shiba:
    """Reference https://medium.com/citycoddee/python%E9%80%B2%E9%9A%8E%E6%8A%80%E5%B7%A7-2-static-class-abstract
    -methods%E4%B9%8B%E5%AF%A6%E7%8F%BE-1e3b3998bccf """
    def __init(self, height, weight):
        self.height = height
        self.weight = weight

    @staticmethod
    def pee(length):
        print("pee" + "." * length)


Shiba.pee(6)
