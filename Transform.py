class ShiftNormalize(object):
    def __init__(self, maxValue):
        self.maxVal = maxValue
        self.midVal = self.maxVal / 2

    def __call__(self, sample):
        return (sample / self.midVal) - 1

    def deNormalize(self, inputs, shift=0):
        return ((inputs + 1) if shift == 0 else inputs) * self.midVal
