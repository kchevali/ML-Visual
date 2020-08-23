
#


class List3D():

    def __init__(self, rows, cols, depth):
        self.rows = rows
        self.cols = cols
        self.depth = depth
        self.n = self.rows * self.cols * self.depth
        self.lengths = [self.cols, self.rows, self.depth]
        self.arr = [self.index(i, j, k) + 1 for k in range(self.depth) for i in range(self.rows) for j in range(self.cols)]

    def index(self, i, j, k):
        return j + self.cols * (i + self.rows * k) if i >= 0 and i < self.rows and j >= 0 and j < self.cols and k >= 0 and k < self.depth else None

    def shift(self, dx=0, dy=0, dz=0):
        d = [dx, dy, dz]
        dd = [d_ if d_ != 0 else 1 for d_ in d]
        t1 = [(1 if d[i] == 1 else (2 - self.lengths[i])) * d[i] for i in range(3)]
        t2 = [t1[i] - d[i] for i in range(3)]
        for i in range(self.rows - abs(dy)):
            for j in range(self.cols - abs(dx)):
                for k in range(self.depth - abs(dz)):
                    a = self.index(t1[1] + i * dd[1], t1[0] + j * dd[0], t1[2] + k * dd[2])
                    b = self.index(t2[1] + i * dd[1], t2[0] + j * dd[0], t2[2] + k * dd[2])
                    self.arr[b] = self.arr[a]
                    if a != b:
                        self.arr[a] = 0
        return self

    def shift(self, dx=0, dy=0, dz=0):
        cx, cy, cz = 0 if dx <= 0 else self.cols - 1, 0 if dy <= 0 else self.rows - 1, 0 if dz <= 0 else self.depth - 1
        ddx, ddy, ddz = -1 if dx > 0 else 1, -1 if dy > 0 else 1, -1 if dz > 0 else 1
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.depth):
                    x, y, z = cx + j * ddx, cy + i * ddy, cz + k * ddz
                    a, b = self.index(y - dy, x - dx, z - dz), self.index(y, x, z)
                    if a != None:
                        if b != None:
                            self.arr[b] = self.arr[a]
                        if a != b:
                            self.arr[a] = 0
                    elif b != None:
                        self.arr[b] = 0
        return self

    # def shift(self, dx=0, dy=0, dz=0):
    #     for i in range(self.rows):
    #         for j in range(self.cols - dx):
    #             for k in range(self.depth):
    #                 a = self.index(i, j + dx, k)
    #                 b = self.index(i, j, k)
    #                 self.arr[b] = self.arr[a]
    #                 self.arr[a] = 0
    #     return self

    def __str__(self):
        out = ""
        for k in range(self.depth):
            out += "=" * 10 + "\n"
            for i in range(self.rows):
                for j in range(self.cols):
                    out += str(self.arr[self.index(i, j, k)]) + " "
                out += "\n"
        return out


if __name__ == '__main__':
    arr = List3D(2, 5, 3)
    print(arr)
    print(arr.shift2(dx=2, dy=1, dz=-1))
    print(arr.shift2(dx=-1, dy=-1))
