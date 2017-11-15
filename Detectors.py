from __future__ import division
import numpy as np
from scipy.stats import gmean, kurtosis
from scipy.signal import fftconvolve, tukey
import random
import math
import time
import scipy


"""Utility functions"""
def st():
    global START_TIME
    START_TIME = time.time()
def et():
    t = time.time() - START_TIME
    print(t)
    return t
def mad(v):
    """
    Implementation by Statsmodel package
    """
    return np.median(np.abs(v-np.median(v)))

def autocorr_matrix(sig):
    autocorr = fftconvolve(sig, np.conj(sig[::-1]))
    autocorr = autocorr[len(autocorr)//2:len(autocorr)//2+64]
    mat = np.array([np.roll(autocorr, i) for i in range(len(autocorr))])
    uidx = np.triu_indices(len(autocorr), 1)
    lidx = np.tril_indices(len(autocorr), -1)
    mat[uidx] = np.conj(mat[uidx])
    mat[lidx] = np.conj(mat.T[lidx])
    return mat

def no_window(x):
    return None

def rect(N):
    return np.ones(N)


def energy(data):
    """Returns energy of signals

    Input
    -----
    data: An array where signals are the rows

    Output
    ------
    A vector where each value correspond to energy of corresponding signal
    """

    return np.real(np.mean(np.abs(data)**2, axis=1))


def avg_fft(data, fft_window, num_fft):
    """Returns an array of averaged ffts

    Input
    -----
    data: An array where signals are the rows
    fft_window: Window to apply to each section of signal
    fft_size: number of samples for each fft

    Output
    ------
    An array where each row corresponds to an averaged fft
    """

    assert len(data[0]) % num_fft == 0, "fft_size not compatible with signal length: fft size-{}, sig_len-{}".format(
        len(data[0]), num_fft)
    fft_size = len(data[0]) // num_fft
    win_split = np.array(np.split(data, num_fft, axis=1))
    if fft_window(1):  # Don't do anything if no window
        win = lambda x: fft_window(fft_size) * x
        win_split = np.apply_along_axis(win, 2, win_split)
    fft_coeffs = np.mean(np.absolute(np.fft.fft(win_split, norm="ortho")) ** 2, axis=0)
    return fft_coeffs


def time_lag(data):
    """Returns value of time delayed correlation divided by the autocorrelation value at index 0

    Input
    -----
    data: An array where signals are the rows
    time_delay: time delay of the correlation

    Output
    ------
    A vector where each value correspond to time shifted index of auto correlation
    """
    normalize = [i for i in range(1,len(data[0])+1)]
    normalize += normalize[-2::-1]
    idx = len(data[0])//4
    def win(sig):
        res = fftconvolve(sig, np.conj(sig)[::-1])/normalize
        res[len(res)//2] = 0
        return res[idx:idx*7]
    autocorr = np.apply_along_axis(win, 1, data)
    return np.max(np.abs(autocorr), axis=1)

def kurtosis_detector(data):
    return np.abs(kurtosis(np.real(data), axis=1) + kurtosis(np.imag(data), axis=1))


def KLT(data):
    """Returns the max eigenvalue of covariance matrix made from data

    Input
    -----
    data: An array where signals are the rows
    time_delay: time delay of the correlation

    Output
    ------
    An array of the max eigenvalue scaled by the trace of the covariance matrices made from data
    """

    eigs = []
    N = 64  # dimension of each outer product matrix
    for sig in data:
        cur_eigs = []
        cov_matrix = np.zeros((N, N), dtype=np.complex128)
        o = np.empty((N, N), dtype=np.complex128)
        for i in range(len(data[0])- N):
            cov_matrix += np.outer(sig[i:i+N], np.conj(sig[i:i+N]), o)
        cov_matrix /= N
        eigv = scipy.linalg.eigvalsh(cov_matrix, eigvals=(63, 63))

        eigs.append(max(eigv) / np.trace(cov_matrix) / N)
    return np.real(np.array(eigs))

def max_KLT(data):
    """Returns value of time delayed correlation

    Input
    -----
    data: An array where signals are the rows
    time_delay: time delay of the correlation

    Output
    ------
    A matrix of eigenvales of the covariance matrix averaged among m_size of each signal
    """

    eigs = []
    p = 64
    N = 64
    for sig in data:
        cov_matrix = autocorr_matrix(sig)
        eigv = np.real(scipy.linalg.eigvalsh(cov_matrix)[::-1])
        eigv = eigv/np.mean(eigv)
        eigs.append(max(eigv))

    return eigs

def MDL_KLT(data):
    """Returns k eigenvalues found using MDL

    Input
    -----
    data: An array where signals are the rows
    time_delay: time delay of the correlation

    Output
    ------
    A matrix of eigenvales of the covariance matrix averaged among m_size of each signal
    """

    eigs = []
    p = 64
    N = len(data[0])//p
    for sig in data:
        splits = np.split(sig, N)
        cov_matrix = np.zeros((p, p), dtype=np.complex128)
        for split in splits:
            split /= np.mean(split)
            cov_matrix += np.outer(split, np.conj(split))

        eigv = np.real(scipy.linalg.eigvalsh(cov_matrix)[::-1])
        eigv = eigv/np.mean(eigv)
        
        best_k = 0
        best_MDL = float("inf")
        for k in range(0,p):
            noise_eigs = eigv[k:]
            noise_dim = len(noise_eigs)
            ratio = gmean(noise_eigs)/np.mean(noise_eigs)
            cur_MDL = -np.log(ratio**(noise_dim*N)) + .5*k*(2*p-k)*np.log(N)
            if cur_MDL < best_MDL:
                best_k = k
                best_MDL = cur_MDL
                
        if best_k == 0:
            eigs.append(0)
        else:
            eigs.append(sum(eigv[:best_k]))
            
    return np.real(np.array(eigs))




def gen_signals(num_signals, sig_len, SNR_dB, sig_type, noise_type):
    """Generates an array of the desired signal with desired SNR
    Inputs
    ------
    num_signals: Amount of sine waves to generate
    sig_len: Length of signals
    SNR_db: Signal to noise ratio of sine waves in dB
    sig_type: The signal type
    noise_type: The noise to add to the signal, if you want to generate noise should be None

    Output
    ------
    A num_signals by sig_len array of noisy signals
    """

    SNR = 10 ** (SNR_dB / 10)
    if sig_type == "sin":
        return gen_complex_sinusoid(num_signals, sig_len, SNR, noise_type)
    elif sig_type == "chirp_narrow":
        return gen_chirps(num_signals, sig_len, SNR, .1/sig_len, noise_type)
    elif sig_type == "chirp_1":
        return gen_chirps(num_signals, sig_len, SNR, 1 / sig_len, noise_type)
    elif sig_type == "chirp_2":
        return gen_chirps(num_signals, sig_len, SNR, 2 / sig_len, noise_type)
    elif sig_type == "noise":
        return gen_noise(num_signals, sig_len)
    elif sig_type == "bpsk":
        return gen_bpsk(num_signals, sig_len, 16, SNR, noise_type)
    elif sig_type == "ham_bpsk":
        return gen_ham_bpsk(num_signals, sig_len, 16, SNR, noise_type)
    elif sig_type == "ar_noise":
        return gen_ar_noise(num_signals, sig_len)


def gen_complex_sinusoid(num_signals, sig_len, SNR, noise_type):
    """Generates an array of noisy sine waves

    Inputs
    ------
    num_signals: Amount of sine waves to generate
    sig_len: Length of signals
    SNR: Signal to noise ratio of sine waves

    Output
    ------
    A num_signals x sig_len array of noisy sine waves
    """

    sins = []
    for i in range(num_signals):
        freq = np.random.rand()
        phase = random.uniform(-np.pi, np.pi)
        ix = np.arange(0, sig_len, 1)
        sin = np.exp(1j * (2 * np.pi * freq * ix + phase))
        sin *= np.sqrt(SNR)
        sins.append(sin)
    sins = np.vstack(sins)
    return sins + gen_signals(num_signals, sig_len, 1, noise_type, None)


def gen_chirps(num_signals, sig_len, SNR, drift_rate, noise_type):
    chirps = []
    for _ in range(num_signals):
        t = np.arange(0, sig_len, 1)
        b = np.random.rand()
        phase = random.uniform(-np.pi, np.pi)
        chirp = np.exp(1j * 2 * np.pi * ((drift_rate * t + b) * t + phase))
        chirp *= np.sqrt(SNR)
        chirps.append(chirp)
    chirps = np.vstack(chirps)
    return chirps + gen_signals(num_signals, sig_len, 1, noise_type, None)


def gen_bpsk(num_signals, sig_len, sample_rate, SNR, noise_type):
    assert sig_len % sample_rate == 0, "Not right signal length for bpsk"
    num_bits = int(sig_len / sample_rate)
    sigs = []
    for i in range(num_signals):
        freq = np.random.rand()
        phase = random.uniform(-np.pi, np.pi)
        ix = np.arange(0, sig_len, 1)
        sin = np.exp(1j * (2 * np.pi * freq * ix + phase))
        sin *= np.sqrt(SNR)
        bits_init = np.random.randint(2, size=num_bits)
        bits_init[bits_init == 0] = -1
        bits = np.zeros(sig_len)
        bits[::sample_rate] = bits_init
        bits = fftconvolve(bits, rect(sample_rate), mode="same")
        sigs.append(sin * bits)

    sigs = np.vstack(sigs)
    return sigs + gen_signals(num_signals, sig_len, 1, noise_type, None)


def gen_ham_bpsk(num_signals, sig_len, sample_rate, SNR, noise_type):
    assert sig_len % sample_rate == 0, "Not right signal length for bpsk"
    num_bits = int(sig_len / sample_rate)
    sigs = []
    for i in range(num_signals):
        freq = np.random.rand()
        phase = random.uniform(-np.pi, np.pi)
        ix = np.arange(0, sig_len, 1)
        sin = np.exp(1j * (2 * np.pi * freq * ix + phase))
        sin *= np.sqrt(SNR)
        bits_init = np.random.randint(2, size=num_bits)
        bits_init[bits_init == 0] = -1
        bits = np.zeros(sig_len)
        bits[::sample_rate] = bits_init
        bits = fftconvolve(bits, np.hamming(sample_rate), mode="same")
        sigs.append(sin * bits)

    sigs = np.vstack(sigs)
    return sigs + gen_signals(num_signals, sig_len, 1, noise_type, None)

def gen_window_bpsk(num_signals, sig_len, sample_rate, SNR):
    
    assert sig_len % sample_rate == 0, "Not right signal length for bpsk"
    num_bits = int(sig_len / sample_rate)
    sigs = []
    for i in range(num_signals):
        freq = .5
#         phase = random.uniform(-np.pi, np.pi)
        phase = 0
        ix = np.arange(0, sig_len, 1)
        sin = np.exp(1j*(2*np.pi*freq*ix+phase))
        #sin *= np.sqrt(2)
        bits_init = np.random.randint(2, size=num_bits)
        bits_init[bits_init == 0] = -1
        bits = np.zeros(sig_len)
        bits[::sample_rate] = bits_init
        bits = fftconvolve(bits, rect(sample_rate), mode="same")
        sigs.append(sin*bits)

    sigs = np.vstack(sigs)
    start = int(sig_len/2 - sig_len/sample_rate)
    end = int(sig_len/2 + sig_len/sample_rate)
    win = np.zeros(sig_len)
    win[start:end] = 1
    win = np.fft.ifft(win)
    fil = lambda x: fftconvolve(x, win, mode="same")
    
    sigs = np.apply_along_axis(fil, 1, sigs)
    norm_sigs = np.apply_along_axis(lambda x: x/np.sqrt(np.var(x))*np.sqrt(SNR), 1, sigs)
    return norm_sigs + gen_noise(num_signals, sig_len)



def gen_noise(num_signals, sig_len):
    """Generates an array of the white noise

    Inputs
    ------
    num_signals: Amount of sine waves to generate
    sig_len: Length of signals

    Output
    ------
    A num_signals by sig_len array of noise
    """

    r_noise = np.random.normal(0, 1, (num_signals, sig_len))
    c_noise = np.random.normal(0, 1, (num_signals, sig_len)) * 1j
    noise = np.add(r_noise, c_noise) / np.sqrt(2)
    return noise/(np.var(noise, axis=1)**.5)[:, None]


def gen_ar_noise(num_signals, sig_len):
    """Generates an array of autoregressive nosie

    Inputs
    ------
    num_signals: Amount of sine waves to generate
    sig_len: Length of signals

    Output
    ------
    A num_signals by sig_len array of noise
    """
    noise = gen_noise(num_signals, sig_len)
    win = lambda x: fftconvolve(x, np.fft.fftshift(np.fft.ifft(tukey(sig_len, .35))), mode="same")
    noise = np.apply_along_axis(win, 1, noise)
    return noise/(np.var(noise, axis=1)**.5)[:, None]


"""Wrapper functions of different settings of detectors"""
def fft_max(data):
    fft_c = avg_fft(data)
    return np.max(fft_c, axis=1)

def fft_mad(data):
    fft_c = avg_fft(data)
    return np.max(fft_c, axis=1)/[mad(v) for v in fft_c]

def fft_amean(data):
    fft_c = avg_fft(data)
    return np.max(fft_c, axis=1)/np.mean(fft_c, axis=1)

def fft_gmean(data):
    fft_c = avg_fft(data)
    return np.max(fft_c, axis=1)/gmean(fft_c, axis=1) # gmean implemented by scipy package

def fft_1(data):
    return np.max(avg_fft(data, no_window, 1), axis=1)

def fft_2(data):
    return np.max(avg_fft(data, no_window, 2), axis=1)

def fft_4(data):
    return np.max(avg_fft(data, no_window, 4), axis=1)

def fft_8(data):
    return np.max(avg_fft(data, no_window, 8), axis=1)

def fft_16(data):
    return np.max(avg_fft(data, no_window, 16), axis=1)

def fft_32(data):
    return np.max(avg_fft(data, no_window, 32), axis=1)

def fft_64(data):
    return np.max(avg_fft(data, no_window, 64), axis=1)

def fft_128(data):
    return np.max(avg_fft(data, no_window, 128), axis=1)

def fft_256(data):
    return np.max(avg_fft(data, no_window, 256), axis=1)

def fft_ham_1(data):
    return np.max(avg_fft(data, np.hamming, 1), axis=1)

def fft_ham_2(data):
    return np.max(avg_fft(data, np.hamming, 2), axis=1)

def fft_ham_4(data):
    return np.max(avg_fft(data, np.hamming, 4), axis=1)

def fft_ham_8(data):
    return np.max(avg_fft(data, np.hamming, 8), axis=1)

def fft_ham_16(data):
    return np.max(avg_fft(data, np.hamming, 16), axis=1)

def fft_ham_32(data):
    return np.max(avg_fft(data, np.hamming, 32), axis=1)

def fft_ham_64(data):
    return np.max(avg_fft(data, np.hamming, 64), axis=1)

def fft_ham_128(data):
    return np.max(avg_fft(data, np.hamming, 128), axis=1)

def fft_ham_256(data):
    return np.max(avg_fft(data, np.hamming, 256), axis=1)

def KLT_1(data):
    return KLT(data, 1)

def time_lag1(data):
    return time_lag(data, 1)

def time_lag10(data):
    return time_lag(data, 10)

def time_lag100(data):
    return time_lag(data, 100)

def time_lag1_norm(data):
    return time_lag(data, 1) / energy(data)

