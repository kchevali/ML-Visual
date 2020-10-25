
class MyLinear:

    def __init__(self, n=1, alpha=0.01, epsilon=0.001, x=[], y=[]):
        self.n = n
        self.cef = [0] * (n + 1)
        self.alpha = alpha
        self.epsilon = epsilon
        self.x = x
        self.y = y
        self.length = len(x)
        self.dJ = self.epsilon

    def fit(self):
        if(self.n == 1):
            if self.dJ >= self.epsilon:
                a = self.cef[0] - self.alpha * self.getJGradient(self.cef[0], self.cef[1])
                b = self.cef[1] - self.alpha * self.getJGradient(self.cef[0], self.cef[1], k=1)
                self.dj = self.getJ(self.cef[0], self.cef[1]) - self.getJ(a, b)
                self.cef = [a, b]
                return True
        return False

    def getJ(self, a, b):
        total = 0
        for i in range(self.length):
            total += (self.y[i] - b * self.x[i] - a)**2
        return total

    def getJGradient(self, a, b, k=0):
        total = 0
        for i in range(self.length):
            total += (b * self.x[i] + a - self.y[i]) * k * self.x[i]
        return total
