class A:

    def __init__(self, a: int):
        self.aa = a
        print(self.aa)
    pass

    def add3(self):
        self.aa += 3
        return "123"


a1 = A(5)
a1.add3()
