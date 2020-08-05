def packed(a, b):
    return (a, b)


def sumNum(a, b):
    return a + b


if __name__ == '__main__':
    print("Sum:", sumNum(*packed(5, 6)))
