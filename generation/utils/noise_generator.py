import numpy as np

class NoiseGenerator:

    def __init__(self):

        pass

    def gaussian(self, size, mean=0, std=1):

        return np.random.normal(loc=mean, scale=std, size=size)

    def repaet_gaussian(self, size, repeats, mean=0, std=1):

        noise = self.gaussian(size=size, mean=mean, std=std).reshape(1, -1)
        return np.repeat(noise, repeats=repeats, axis=0)

    def gaussian_variating(self, T, F, size, mean=0, std=1, allow_indentical=False):
        """Gnerate Gaussian noise of time T with variating interval F

        Args:
            allow_identical -- If True, 0.5 probability that the following interval is identical to previous interval
            dim -- dimension of noise
        """

        K = int(np.ceil(T / F)) - 2

        noise = self.repaet_gaussian(size=size, repeats=F, mean=mean, std=std)

        if T <= F:
            return self.repaet_gaussian(size=size, repeats=T, mean=mean, std=std)

        for k in range(K):

            if np.random.rand() > 0.5:

                noise = np.concatenate([noise, noise[-F:]], axis=0)

            else:

                noise = np.concatenate([noise, self.repaet_gaussian(
                    size=size, repeats=F, mean=mean, std=std)], axis=0)

        sup_t = T - noise.shape[0]

        if np.random.rand() > 0.5:

            noise = np.concatenate([noise, noise[-sup_t:]], axis=0)

        else:

            noise = np.concatenate([noise, self.repaet_gaussian(
                size=size, repeats=sup_t, mean=mean, std=std)], axis=0)

        return noise


if __name__ == "__main__":

    ng = NoiseGenerator()

    n = ng.gaussian_variating(T=5, F=2, size=1, allow_indentical=True)

    print(n)
