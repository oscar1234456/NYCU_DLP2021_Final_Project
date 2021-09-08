class ShiftNormalize(object):
    def __init__(self, maxValue):
        self.maxVal = maxValue
        self.midVal = self.maxVal / 2

    def __call__(self, sample):
        return (sample / self.midVal) - 1

    def deNormalize(self, input):
        return (input + 1) * self.midVal