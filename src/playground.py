count = 1


def run():
    global count
    arr = [10] * count
    count += 1
    return arr


if __name__ == '__main__':
    for i in run():
        print(i)
