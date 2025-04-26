import math
import random
import numpy
import matplotlib.pyplot as plt
import pandas as pd


class ScaledSymmetricRandomWalkModel:

    UP_MOVE = 1
    DOWN_MOVE = -1

    def __init__(self, scale_factor: int, total_time):
        self.scale_factor = scale_factor
        self.T = total_time
        self.scaled_delta_t = math.sqrt(float(self.T / self.scale_factor))

    def __compute_initial_walk__(self):
        s = numpy.zeros(self.T + 1)
        for t in range(1, self.T + 1):
            s[t] = s[t-1] + ScaledSymmetricRandomWalkModel.next_random_move()

        return s

    @classmethod
    def next_random_move(cls):
        prob = random.random()
        if prob >= 0.5:
            return ScaledSymmetricRandomWalkModel.UP_MOVE
        return ScaledSymmetricRandomWalkModel.DOWN_MOVE

    def __interpolate_step__(self, walk, n_t, sq_root_scale_factor):
        t = float(n_t / self.scale_factor)
        t_upper = int(math.ceil(t))
        t_lower = int(math.floor(t))
        s_t_upper = walk[t_upper] / sq_root_scale_factor
        s_t_lower = walk[t_lower] / sq_root_scale_factor
        return float(s_t_lower + float((((t - t_lower) / (t_upper - t_lower)) * (s_t_upper - s_t_lower))))

    def __compute_scaled_walk__(self, initial_walk):
        scaled_walk = pd.DataFrame(columns=['t', 'S'])
        n_t = 0.0
        sq_root_scale_factor = math.sqrt(self.scale_factor)
        while n_t <= (self.T * self.scale_factor):
            if n_t.is_integer():
                scaled_s = float(
                    initial_walk[int(math.floor(n_t))] / sq_root_scale_factor)
            else:
                scaled_s = self.__interpolate_step__(
                    initial_walk, n_t, sq_root_scale_factor)

            scaled_walk = pd.concat(
                [scaled_walk, pd.DataFrame([{'t': n_t, 'S': scaled_s}])], ignore_index=True)

            n_t += self.scaled_delta_t

        return scaled_walk

    def plot_scaled_walk(self):
        initial_walk = self.__compute_initial_walk__()
        scaled_walk = self.__compute_scaled_walk__(initial_walk)
        self.__plot__(initial_walk, scaled_walk)

    def __plot__(self, initial_walk, scaled_walk):
        plt.style.use("seaborn-v0_8")
        _, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(initial_walk[:self.T], label='Random Walk : ' +
                   str(self.T) + ' Random Variables')
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('S')
        ax[0].legend()

        ax[1].plot(scaled_walk['t'], scaled_walk['S'], label='Scaled Random Walk :' +
                   str(self.T * self.scale_factor) + ' Random Variables')
        ax[1].set_xlabel(str(self.scale_factor) + 't')
        ax[1].set_ylabel('S')
        ax[1].legend()

        plt.show()
