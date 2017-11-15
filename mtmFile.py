import numpy as np 
import matplotlib.pyplot as plt

class mtmFile:

    def __init__(self, filename):
        with open(filename, "rb") as f:
        [self.center_freq, self.bandwidth, self.num_timesteps, self.num_freqs] = np.fromfile(f, dtype=np.float64, count=4)
        self.num_chans = 64
        self.samples_per_unit = self.num_freqs / self.num_chans  #  Unit is a bloc channel

    def get_data(self, timestep, channel):
        loc = timestep * self.num_freqs + channel * self.samples_per_unit
        loc += 64 / 8 * 4  # Offset for header
        with open(filename, "rb") as f:
            f.seek(loc)
            data = np.fromfile(f, dtype=np.int8, count=self.samples_per_unit)
        return data

    def plot_coarse_channel(self, channel):
        data = self.get_data(self, 0, channel):
        for i in range(1, self.num_timesteps):
            data = np.vstack((data, self.get_data(i, channel)))

        lfreq, ufreq = self._calc_chan_freq_range(chan)
        freq_axis = np.linspace(lfreq, ufreq, len(data[0]))
        plt.imshow(data, extent=[lfreq, ufreq, 0, self.num_timesteps])

    def _calc_chan_freq_range(self, chan):
        """calculates the range of frequencies contained in a channel"""

        chan_bw = self.bandwidth / self.num_chans
        calc_from = self.center_freq - (self.bandwidth / 2)  # find freq associated with 0th channel
        cur_chan_range = [calc_from + chan * chan_bw, calc_from + (chan + 1) * chan_bw]
        cur_chan_range.sort()  # bandwidth could be negative which would have higher number first
        min_freq, max_freq = cur_chan_range[0], cur_chan_range[1]
        return min_freq, max_freq


