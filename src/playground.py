class A:
    def __init__(self):
        self.a = "a"


class B(A):
    def __init__(self):
        self.a = "a"


if __name__ == '__main__':
    a = A()
    b = B()
    print(issubclass(A, type(a)))
    print(issubclass(type(b), A))
    print(issubclass(B, type(b)))
