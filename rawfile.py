import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
"""Utility class for reading raw file data"""

class RawFile:
    """Class to manage withdrawing information and analyzing it from raw files. Does not actually store any data
    within itself
    Note: blocs are 0 indexed, only works if there are two polarizations
    """

    def __init__(self, filename):
        self.filename = filename
        self.bloc_size = 0
        self.num_chans = 0
        self.chan_size = 0
        self.header_size = 0
        self.num_blocs = 0
        self.center_freq = 0
        self.bandwidth = 0
        self.bytes_per_sample = 0
        self.num_pol = 0  # number of polarization 
        self.num_bits = 0  # how many bits make up real/imag portion of sample. Therefore 1 sample is twice as large as this number
        self.samples_per_bloc_chan = 0  # amount of samples stored in 1 bloc in 1 channel and 1 polarization
        self.set_file_info(filename)  # This function sets the above variables

    def set_file_info(self, filename):
        """Sets up some class variables that are useful

        Input
        -----
        filename: Name of file that RawFileModel belongs to
        """
        # open the file at the beginning
        with open(filename, 'r') as f:

            # store nChanSize, nBlocsize, and nHeadersize of the file

            # read the header of the first bloc 80 characters at a time
            currline = f.read(80)  # read 1st line
            nHeadLine = 1  # counter for number of lines in header to store header size

            # read header
            while (not 'END  ' in currline):

                # store nBlocsize
                if ('BLOCSIZE' in currline):
                    subline = currline[10:]  # remove keyword from string
                    subline.replace(' ', '')  # remove empty space
                    self.bloc_size = int(subline)  # convert string to integer

                if 'NBITS' in currline:
                    subline = currline[10:]  # remove keyword from string
                    subline.replace(' ', '')  # remove empty space
                    self.num_bits = int(subline)  # convert string to integer

                if 'NPOL' in currline:
                    subline = currline[10:]  # remove keyword from string
                    subline.replace(' ', '')  # remove empty space
                    self.num_pol = int(subline)  # convert string to integer
                    if self.num_pol > 2: self.num_pol = 2

                # store nChan
                if ('OBSNCHAN' in currline):
                    subline = currline[10:]  # remove keyword from string
                    subline.replace(' ', '')  # remove empty space
                    self.num_chans = int(subline)  # convert string to integer

                # store information from DIRECTIO necessary to computer nHeadSize
                if ('DIRECTIO' in currline):
                    subline = currline[10:]  # remove keyword from string
                    subline.replace(' ', '')  # remove empty space
                    directio = int(subline)  # convert string to integer

                if ('OBSFREQ' in currline):
                    subline = currline[10:]  # remove keyword from string
                    subline.replace(' ', '')  # remove empty space
                    self.center_freq = float(subline)  # convert string to integer

                if ('OBSBW' in currline):
                    subline = currline[10:]  # remove keyword from string
                    subline.replace(' ', '')  # remove empty space
                    self.bandwidth = float(subline)  # convert string to integer

                currline = f.read(80)
                nHeadLine = nHeadLine + 1  # count number of lines in header

        self.chan_size = int(self.bloc_size / self.num_chans)  # size of 1 coarse channel

        if directio == 1:
            self.header_size = 512 * (math.floor((nHeadLine * 80) / 512.) + 1)  # size of header
        if directio == 0:
            self.header_size = nHeadLine * 80  # size of header

        self.num_blocs = int(np.ceil(os.path.getsize(filename) / (self.header_size + self.bloc_size)))
        self.bytes_per_sample = self.num_pol * 2 * self.num_bits // 8
        self.samples_per_bloc_chan = int(self.bloc_size / self.bytes_per_sample / self.num_chans / self.num_pol)

    def get_data(self, bloc, chan, num_samples=None):
        """Retrieves data from raw file depending on parameters. Two modes depending on type of
        num_samples. Only can work inside a bloc-channel

        Inputs
        ------
        bloc: bloc to retrieve sample from
        chan: Channel to retrieve samples from
        num_samples: If num samples is an integer will return that amount of samples starting from
        the beginning of the bloc.
        ex. if num_samples = 50, will return first 50 samples. 

        Output
        ------
        pol0: Array of complex numbers corresponding to 0th polarization
        pol1: Array of complex numbers corresponding to 1st polarization
        """

        num_samples = num_samples or self.samples_per_bloc_chan

        # compute where in the file to start reading
        nSkip = int(bloc * (self.header_size + self.bloc_size) + self.header_size + (chan * self.chan_size))
        with open(self.filename, "rb") as f:
            f.seek(nSkip, 0)
            sig = np.fromfile(f, dtype='int8', count=num_samples * self.bytes_per_sample)

        # This line sort of hard to understand
        # pol0, pol1 = eval("self._get_data_{}(sig)".format(self.num_bits))

        if self.num_bits == 2:
            pol0, pol1 = self._get_data_2(sig)

        elif self.num_bits == 4:
            pol0, pol1 = self._get_data_4(sig)

        elif self.num_bits == 8:
            pol0, pol1 = self._get_data_8(sig)

        return np.array(pol0), np.array(pol1)

    def _get_data_2(self, sig):
        """Gets data organized with each byte containing two complex samples"""
        re0 = sig >> 6
        im0 = sig >> 4 & 0x3
        re1 = sig >> 2 & 0x3
        im1 = sig & 0x3
        return np.add(re0, im0 * 1j), np.add(re1, im1 * 1j)

    def _get_data_4(self, sig):
        """Gets data organized with each byte containing one complex samples"""
        pol0 = sig[::2]
        pol1 = sigh[1::2]
        re0 = pol0 >> 4
        im0 = pol0 & 0xF
        re1 = pol1 >> 4
        im1 = pol1 & 0xF
        return np.add(re0, im0 * 1j), np.add(re1, im1 * 1j)

    def _get_data_8(self, sig):
        """Gets data organized with each byte containing half a complex samples"""
        re0 = sig[0::4]
        im0 = sig[1::4]
        re1 = sig[2::4]
        im1 = sig[3::4]
        return np.add(re0, im0 * 1j), np.add(re1, im1 * 1j)

    def plot_sxx(self, bloc, chan, num_samples=None):
        """Returns the spectral energy of a signal and plots it

        Input
        -----
        bloc: Bloc signal comes from
        chan: Channel signal comes from
        num_samples: Number of samples signal should consist of

        Outputs
        -------
        sxx0: Spectral energy from 0th polarization
        sxx1: Spectral energy from 1st polarization
        """

        pol0, pol1 = self.get_data(bloc, chan, num_samples)
        sxx0, sxx1 = np.fft.fftshift(np.abs(np.fft.fft(pol0)) ** 2), np.fft.fftshift(np.abs(np.fft.fft(pol1)) ** 2)
        plt_title = "SPD of bloc: {}, channel {}".format(str(bloc), str(chan))
        lfreq, ufreq = self._calc_chan_freq_range(chan)  # get lower bound and upper bound frequency of the channel
        freq_axis = np.linspace(lfreq, ufreq, len(sxx0))
        self._plot(sxx0, sxx1, plt_title, x=freq_axis)
        return sxx0, sxx1

    def plot_log_sxx(self, bloc, chan, num_samples=None):
        """Returns the log spectral energy of a signal and plots it

        Input
        -----
        bloc: Bloc signal comes from
        chan: Channel signal comes from
        num_samples: Number of samples signal should consist of

        Outputs
        -------
        sxx0: Log spectral energy from 0th polarization
        sxx1: Log spectral energy from 1st polarization
        """

        pol0, pol1 = self.get_data(bloc, chan, num_samples)
        sxx0, sxx1 = np.abs(np.fft.fft(pol0)) ** 2, np.abs(np.fft.fft(pol1)) ** 2
        log_sxx0, log_sxx1 = np.fft.fftshift(10 * np.log10(sxx0)), np.fft.fftshift(10 * np.log10(sxx1))
        plt_title = " Log SPD of bloc: {}, channel {}".format(str(bloc), str(chan))
        lfreq, ufreq = self._calc_chan_freq_range(chan)  # get lower bound and upper bound frequency of the channel
        freq_axis = np.linspace(lfreq, ufreq, len(log_sxx0))
        self._plot(log_sxx0, log_sxx1, plt_title, x=freq_axis)
        return log_sxx0, log_sxx1

    def avg_spectra(self, bloc, chan, fft_size, num_spectra, start_sample, plot=True):
        """Returns and plots an averaged log SPD

        Inputs
        ------
        bloc: Bloc data is from
        chan: Channel data is from
        fft_size: Size of the fft of each spectra
        num_spectra: Number of spectra to be averaged over
        start_sample: Start sample in the bloc
        plot: Whether or not to plot

        Outputs
        -------
        avg_sxx0: Averaged sxx of polarization 0
        avg_sxx1: Averaged sxx of polatization 1
        """

        # computations to see how to get data
        samples_needed = fft_size * num_spectra
        end_sample = start_sample + samples_needed

        data0, data1 = self.get_data(bloc, chan, [start_sample, end_sample])
        fft_ready0 = np.reshape(data0, (num_spectra, fft_size))
        fft_ready1 = np.reshape(data1, (num_spectra, fft_size))
        sxx0 = 10 * np.log10((np.abs(np.fft.fft(fft_ready0)) ** 2))
        sxx1 = 10 * np.log10((np.abs(np.fft.fft(fft_ready1)) ** 2))
        avg_sxx0 = np.mean(sxx0, axis=0)
        avg_sxx1 = np.mean(sxx1, axis=0)
        avg_sxx0[0], avg_sxx1[0] = np.mean(avg_sxx0), np.mean(avg_sxx1)  # 0 out spike at DC

        avg_sxx0 = np.fft.fftshift(avg_sxx0)
        avg_sxx1 = np.fft.fftshift(avg_sxx1)
        lfreq, ufreq = self._calc_chan_freq_range(chan)  # get lower bound and upper bound frequency of the channel
        freq_axis = np.linspace(lfreq, ufreq, fft_size)

        if plot:
            plt_title = "Log PSD: {} spectra, {} samples, bloc/chan {}/{}".format(
                str(num_spectra), str(fft_size), str(bloc), str(chan))

            self._plot(avg_sxx0, avg_sxx1, plt_title, x=freq_axis)

        return avg_sxx0, avg_sxx1

    def spectrogram(self, bloc, chan, fft_size, num_spectra, time_steps=None, plot=True):
        """
        returns and optionally plots a spectrogram of the given data. Each time step is an
        averaged fft in dB

        Inputs
        ------
        bloc: Bloc data is located in
        chan: Channel of data
        fft_size: Size off fft of the spectra
        num_spectra: Number of spectras that each time step should be averaged over
        time_steps: Number of time steps. If none will attempt to plot entire bloc
        plot: Whether or not to plot a spectrogram using matplotlib

        Output
        ------
        spectrogram: Final spectrogram
        """

        # calculate how many blocs you'll need
        # get data by bloc to save memory
        step_size = fft_size * num_spectra
        time_steps = time_steps or int(self.samples_per_bloc_chan // (fft_size * num_spectra))
        cur_sample = 0
        spec0, spec1 = np.zeros((fft_size, time_steps)), np.zeros((fft_size, time_steps))
        for i in range(time_steps):
            ts0, ts1 = self.avg_spectra(bloc, chan, fft_size, num_spectra, cur_sample, False)
            spec0[:, i], spec1[:, i] = ts0, ts1
            cur_sample = cur_sample + step_size

        if plot:
            lfreq, ufreq = self._calc_chan_freq_range(chan)
            freq_axis = np.linspace(lfreq, ufreq, fft_size)
            fig, axarr = plt.subplots(1, 2, sharey=True)
            axarr[0].pcolormesh(np.arange(0, time_steps, 1), freq_axis, spec0, cmap="viridis")
            axarr[0].set_title("Polarization 0")
            im = axarr[1].pcolormesh(np.arange(0, time_steps, 1), freq_axis, spec1, cmap="viridis")
            axarr[1].set_title("Polarization 1")
            cbaxes = fig.add_axes([0.93, 0.1, 0.02, 0.77])
            cb = plt.colorbar(im, cax=cbaxes)
            title = "Spectrogram of bloc/chan {}/{}".format(bloc, chan)
            fig.suptitle(title, fontsize=16)
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            fig.subplots_adjust(right=.91)
            ax = plt.gca()
        # ax.get_yaxis().get_major_formatter().set_useOffset(False)
        return spec0, spec1

    def convert_to_mtm(self, p, num_tapers):
        """Converts the Raw File to mtmf"""
        from spectrum import dpss
        import multiprocessing as mp

        print("num bits: " + str(self.num_bits))
        print("Generating tapers")
        tapers, eigs = dpss(self.samples_per_bloc_chan, p, num_tapers)
        print("Tapers generated")

        filename = self.filename[:-4] + ".mtmf"

        
        pool = mp.Pool(processes=8)
        cur_max_mag = 0
        with open("temp.tmp", "wb") as tmp:
            for bloc in range(self.num_blocs):
                print("computing max " +str(bloc))
                start = time.time()
                half_data = [self.get_data(bloc, chan)[0] for chan in range(self.num_chans)]
                
                results = pool.map(mtm_method, [(half_data[i], tapers, eigs, i) for i in range(self.num_chans)])
                results = sorted(results, key=lambda x: x[2])
                print(len(results))
                print(len(results[0][0]), self.samples_per_bloc_chan)
               # print(results[0][0][:10])
               
                for result in results: 
                    result[0].tofile(tmp)
                    cur_max_mag = max(cur_max_mag, result[1])

                print("bloc time: " + str(time.time()-start))

        bins = np.linspace(0, cur_max_mag, 257)

        samples_written = 0
        num_samples = 0
        print(self.num_chans, self.num_blocs)
        print("Converting data")
        start = time.time()
        base_file = os.path.basename(filename)
        with open(base_file, "wb") as writef, open("temp.tmp", "rb") as readf:
            np.array([self.center_freq, self.bandwidth, self.num_blocs, self.samples_per_bloc_chan*self.num_chans]).tofile(writef)
            bloc_chan_size = self.samples_per_bloc_chan * 8
            for i in range(self.num_blocs*self.num_chans):
                readf.seek(i * bloc_chan_size)
                mtm_data = np.fromfile(readf, dtype=np.float64, count=self.samples_per_bloc_chan)
                num_samples += len(mtm_data)
                print(mtm_data[0:10])
                quantized = np.digitize(mtm_data, bins)  # Bins start from 1
                quantized -= 129
                quantized = quantized.astype(np.int8)
                samples_written += len(quantized)
                quantized.tofile(writef)

        print("time to convert: " + str(time.time()-start))
        print(samples_written, num_samples)
        print(self.samples_per_bloc_chan*128*64)
        os.remove("temp.tmp")

    def write_header(self, filename):
        with open(filename, "ab") as fw, open(self.filename, "rb") as fr:
            data = fr.read(self.header_size)
            fw.write(data)

    def write_blocks(self, filename, start_bloc, end_bloc):
        """
        Writes certain blocks of raw file to a new file


        Inputs
        ------
        filename: Name of file to write data to
        start_block: Block to start writing from
        end_block: Block to end at. Does not include this block

        """

        start_byte = start_bloc * (self.header_size + self.bloc_size)
        write_bytes = (end_bloc - start_bloc) * (self.header_size + self.bloc_size)
        with open(filename, "ab") as fw, open(self.filename, "rb") as fr:
            fr.seek(start_byte)
            data = fr.read(write_bytes)
            fw.write(data)

    def write_bloc_channel(self, filename, bloc, channel):

        # Writes a bloc_channel to a file

        with open(filename, "ab") as fw, open(self.filename, "rb") as fr:
            nSkip = int(bloc * (self.header_size + self.bloc_size) + self.header_size + (chan * self.chan_size))
            fr.seek(nSkip)
            data = fr.read(self.chan_size)
            fw.write(data)

    def write_channel(self, filename, channel):

        # Write an entire channel to a file

        for bloc in range(self.num_blocs):
            self.write_bloc_channel(filename, bloc, channel)

    def rewrite_bloc_channel(self, bloc, chan, sig):
        
        assert len(sig) == 4*self.samples_per_bloc_chan

        # compute where in the file to start reading
        nSkip = int(bloc * (self.header_size + self.bloc_size) + self.header_size + (chan * self.chan_size))
        with open(self.filename, "r+b") as f:
            f.seek(nSkip, 0)
            f.write(sig)

    def rewrite_channel(self, chan):

        # Hard coded. Would not recommend using

        N = self.samples_per_bloc_chan * self.num_blocs
        x = np.arange(0,N,1)
        f = np.cos(np.linspace(-8*np.pi,8*np.pi,N))*4./N + 32./N
        sig = np.exp(f*np.pi*1j*x)
        #plt.plot(sig)
        #plt.show()
        real = np.real(sig)
        imag = np.imag(sig)  
        gmin = min(np.min(real), np.min(imag))
        gmax = max(np.max(real), np.max(imag))
        bins = np.linspace(gmin, gmax, 257)
        quantized_real = np.digitize(real, bins)  # Bins start from 1
        quantized_real -= 129
        quantized_real = quantized_real.astype(np.int8)
        quantized_imag = np.digitize(imag, bins)  # Bins start from 1
        quantized_imag -= 129
        quantized_imag = quantized_imag.astype(np.int8)
        quantized = np.zeros(4*N, dtype=np.int8)
        #plt.plot(quantized_real)
        #plt.show()
        quantized[::4] = quantized_real
        quantized[1::4] = quantized_imag

        for i in range(self.num_blocs):
            M = 4 * self.samples_per_bloc_chan
            write_this = quantized[i*M:i*M + M]
            self.rewrite_bloc_channel(i, chan, write_this)



    def export_spectrograms(self, blocs, fft_size, num_spectra, time_steps):
        """
        Saves generated spectograms to a new folder

        """
        result_dir = "spec_" + self.filename[:-4]
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for bloc in blocs:
            for chan in range(2):
                f.spectrogram(bloc, chan, fft_size, num_spectra, time_steps)
                plt_name = "bloc_{}_chan_{}".format(bloc, chan)
                plt.savefig(result_dir + "/" + plt_name)

    def _plot(self, pol0, pol1, title, x=None):
        """Helper function to plot the various data"""
        if x == None:
            x = np.arange(0, len(pol0), 1)
        fig, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(x, pol0)
        axarr[0].set_title('Polarization 0')
        plt.xlim([x[0], x[-1]])
        axarr[1].plot(x, pol1)
        axarr[1].set_title('Polarization 1')
        plt.xlim([x[0], x[-1]])
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        # turn off scientific notation for axis ticks
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)

    def _calc_chan_freq_range(self, chan):
        """calculates the range of frequencies contained in a channel"""

        chan_bw = self.bandwidth / self.num_chans
        calc_from = self.center_freq - (self.bandwidth / 2)  # find freq associated with 0th channel
        cur_chan_range = [calc_from + chan * chan_bw, calc_from + (chan + 1) * chan_bw]
        cur_chan_range.sort()  # bandwidth could be negative which would have higher number first
        min_freq, max_freq = cur_chan_range[0], cur_chan_range[1]
        return min_freq, max_freq


def mtm_method(args):
    sig, tapers, eigs, i = args
    N = len(sig)
    mtm = np.zeros(N)
    for taper, eig in zip(tapers.T, eigs):
        mtm += eig*np.abs(np.fft.fft(sig * taper))**2
    res = mtm/sum(eigs)
    return res, np.max(res), i
