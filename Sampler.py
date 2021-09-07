import numpy as np
import torch
class Sampler():
    def __init__(self, sampleAmount=1, sampleSize=30, sampleDistribution="Normal", block="Rare+"):
        '''
        :param sampleAmount: The number of samples for each time
        :param sampleSize: The dimension of sample vector
        :param sampleDistribution: The distribution which will sample from
        :param block: Specified range in distribution bulk
        '''
        distributionSampler = {
            "Normal":  np.random.normal, # Sample from Normal distribution
        }
        sampleRange = {
            "Rare+": (1.0, 1.3),
            "NotRare+": (0.85, 1.0),
            "VeryCommon+": (0.75, 0.85),
            "VeryCommon-": (0.65, 0.75),
            "NotRare-": (0.5, 0.65),
            "Rare-": (0.3, 0.5)
        }

        self.sampleDistribution = distributionSampler[sampleDistribution]
        self.sampleAmount = sampleAmount
        self.sampleSize = sampleSize
        self.sampleRange = sampleRange[block]

    def sample(self):
        selectedList = list()
        print(">>>Start Sampling")
        complete = False
        while not complete:
            candidates = self.sampleDistribution(0, 1, (self.sampleAmount*1000, 30))
            i = 0
            for candidate in candidates:
                if self.sampleRange[0] <= self._caulateSTD(candidate) <= self.sampleRange[1]:
                    print("Find!")
                    # print(candidate)
                    # print(np.std(candidate))
                    selectedList.append(candidate)
                    i += 1
                    if self.sampleAmount == i:
                        complete = True
                        break

        print(">>>End Sampling")
        # print(np.mean(candidate[2]))
        # print(np.std(candidate[2]))
        # print(candidate[2])
        return torch.Tensor(selectedList)

    def _caulateSTD(self, latentVector, mean=0):
        return np.sqrt(np.mean(np.square(latentVector-mean)))


if __name__ == "__main__":
    sampler = Sampler(1,"Normal")
    latent_tensor = sampler.sample()
    print("")