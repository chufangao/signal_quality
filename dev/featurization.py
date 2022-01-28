import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
from datetime import datetime
import pickle
from scipy.signal import find_peaks
import warnings
from scipy.stats import skew

import biosppy
import pyhrv
# import entropy
import antropy

from importlib import reload

class Featurize:
    def __init__(self, ecg, pleth, N=5, sampling_rate=125., threshold=.48, 
                 rpeaks_ratio=2., nni_delta=100):
        """
        ecg: a DataFrame, the first column is time, the second column is ECG signal value
        pleth: a DataFrame, the first column is time, the second column is pleth signal value
        N: Number of samples to calculate moving averages for smoothing
        sampling_rate: sampling rate of ECG signal
        threshold: threshold for detecting ECG peaks
        rpeaks_ratio: parameter to determine abnormal r peaks
        nni_delta: parameter to remove abnormal NNIs
        """
        # Smoothen Pleth using moving average
        self.ecg = ecg.value.to_numpy()
        self.ecg_time = ecg.time.to_numpy()
        self.pleth = np.convolve(pleth['value'], np.ones(N)/N, mode = 'valid') 
        self.pleth_time = pleth.time.to_numpy()
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.rpeaks_ratio = rpeaks_ratio
        self.nni_delta = nni_delta
        
    def get_all_features(self):
        # if features cannot be extracted successfully from this segment
        # return NaNs
        self.features = np.array([np.nan]*31)
        self.feature_names = ['']*31
        
        self.segment_pleth()
        if len(self.beat_start) > 0:
            self.find_systolic_peaks()
        
        self.find_r_peaks()
        if len(self.r_peaks) > 0:
            self.get_nni()
        
        if len(self.beat_start) > 0 and len(self.r_peaks) > 0:
            mintime = max(self.systolic_peaks_time[0], self.r_peaks_time[0])
            maxtime = min(self.systolic_peaks_time[-1], self.r_peaks_time[-1])
            # the overlapping duration between pleth and ECG >= 1 min
            if (maxtime - mintime) >= 60: 
                self.find_upstroke_time()
                self.get_skewness_beats()
                self.find_pulse_interval()
                self.find_peak_to_peak_interval()
                self.get_feature_aggregates()
                self.get_ecg_features()
                self.get_ptt()
                
                self.features = np.concatenate((
                    self.pleth_features, self.ecg_features, self.ptt_features))
                self.feature_names = self.pleth_feature_names + self.ecg_feature_names + \
                self.ptt_feature_names
                
        return self.features, self.feature_names 
    
    def get_ptt(self):
        """
        Compute PTT as the time difference between pleth peak and ECG peak.
        """
        systolic_peaks_time = self.systolic_peaks_time
        # pleth peak should be after the r peak
        systolic_peaks_time = systolic_peaks_time[systolic_peaks_time > self.r_peaks_time[0]]
        
        ptt = []
        for i in range(len(systolic_peaks_time)):
            r_peaks_time = self.r_peaks_time
            r_peaks_time = r_peaks_time[r_peaks_time < systolic_peaks_time[i]]
            t = systolic_peaks_time[i] - r_peaks_time[-1]
            k = 1
            while t <= 0.2 and k + 1 <= len(r_peaks_time):
                t = systolic_peaks_time[i] - r_peaks_time[-1-k]
                k += 1
            if t > 0.2:
                ptt.append(t)
        
        if len(ptt) > 0:
            self.ptt = np.asarray(ptt)
            ptt_median = np.median(self.ptt)
            ptt_iqr = np.quantile(self.ptt, q=0.75) - np.quantile(self.ptt, q=0.25)
            ptt_slope = np.polyfit(x = np.arange(self.ptt.shape[0]), y=self.ptt, deg=1)[0]
            self.ptt_features = np.array([ptt_median, ptt_iqr, ptt_slope])
        else:
            # we will have to drop this example if no ptt is valid during the entire 2 min
            self.ptt_features = np.array([np.nan, np.nan, np.nan])
        self.ptt_feature_names = ['PTT_median', 'PTT_IQR', 'PTT_slope']
        
    def segment_pleth(self):
        """
        Segments the Pleth waveform into beats. 
        """
        # The beats are at least 50 samples away
        beat_start, _ = find_peaks(-1 * self.pleth, distance = 50)
        if len(beat_start) > 0:
            beat_end = np.roll(beat_start, -1) # Roll start arrays by one before
            self.beat_start = beat_start[:-1]
            self.beat_end = beat_end[:-1]
            self.beat_start_time = self.pleth_time[self.beat_start]
            self.beat_end_time = self.pleth_time[self.beat_end]
        else:
            # beat_start and beat_end are empty
            self.beat_start = beat_start
            self.beat_end = beat_start
            self.beat_start_time = beat_start
            self.beat_end_time = beat_start

    def find_systolic_peaks(self):
        self.systolic_peaks, _ = find_peaks(self.pleth[self.beat_start[0] : self.beat_end[-1]], \
        distance = 50) # The beats are at least 50 samples away. 
        self.systolic_peaks = self.systolic_peaks + self.beat_start[0]
        # Only compute beats from the start of the first beat to the end of the last beat
        # The amplitude is y-coordinate of the signal at the position of the peak
        self.systolic_peaks_time = self.pleth_time[self.systolic_peaks]
        self.systolic_amplitudes = self.pleth[self.systolic_peaks] 
        
    def find_upstroke_time(self):
        """
        We define it as the time from the start of the beat to its systolic peak.
        If beat_start and systolic_peaks are not perfectly 1-on-1 aligned, 
        consider the closest systolic peak to the beat start time instead.
        """
        try:
            self.upstroke_time = self.systolic_peaks_time - self.beat_start_time
        except: # Sometimes there is a mismatch between the number of systolic peaks and beats
            upstroke_time = []
            systolic_peaks = []
            systolic_amplitudes = []
            for i in range(len(self.beat_start)):
                systolic_peak = self.systolic_peaks[self.systolic_peaks > self.beat_start[i]]
                if len(systolic_peak) > 0: # Due to some issues, the last beat may not have an associated systolic peak
                    systolic_peak = systolic_peak[0] # Consider the closest systolic peak to the beat start time
                    amp_at_times = self.pleth[systolic_peak]
                    systolic_peaks.append(systolic_peak)
                    systolic_amplitudes.append(amp_at_times)
                    upstroke_time.append(self.pleth_time[systolic_peak] - self.pleth_time[self.beat_start[i]])

            self.systolic_peaks = np.asarray(systolic_peaks)
            self.systolic_peaks_time = self.pleth_time[self.systolic_peaks]
            self.systolic_amplitudes = np.asarray(systolic_amplitudes)
            self.upstroke_time = np.asarray(upstroke_time)

    def find_peak_to_peak_interval(self):
        peak_to_peak_interval = self.systolic_peaks_time - np.roll(self.systolic_peaks_time, 1)
        peak_to_peak_interval[0] = peak_to_peak_interval[1]
        self.peak_to_peak_interval = peak_to_peak_interval

    def find_pulse_interval(self):
        pulse_interval = self.beat_start_time - np.roll(self.beat_start_time, 1)
        pulse_interval[0] = pulse_interval[1]
        self.pulse_interval = pulse_interval

    def get_skewness_beats(self):
        self.beat_skewness = np.asarray([skew(self.pleth[self.beat_start[i]:self.beat_end[i]]) for i in range(self.beat_start.shape[0])])

    def get_ts_features(self):
        """ Concatenates all the raw time series features computed for this segment """
        return np.concatenate([self.systolic_amplitudes.reshape(-1, 1), self.peak_to_peak_interval.reshape(-1, 1), 
            self.pulse_interval.reshape(-1, 1), self.upstroke_time.reshape(-1, 1), self.beat_skewness.reshape(-1, 1)], axis = 1)

    def get_feature_aggregates(self):
        ts_feature_names = ['Pleth_systolic_amplitudes', 'Pleth_peak_to_peak_interval', 'Pleth_pulse_interval',
                            'Pleth_upstroke_time', 'Pleth_beat_skewness']
  
        self.pleth_features = np.array([])
        self.pleth_feature_names = []
        for i, ts_features in enumerate([self.systolic_amplitudes, self.peak_to_peak_interval, 
                                         self.pulse_interval, self.upstroke_time, self.beat_skewness]):
            median = np.median(ts_features)
            IQR = np.quantile(ts_features, q=0.75) - np.quantile(ts_features, q=0.25)
            slope = np.polyfit(x = np.arange(ts_features.shape[0]), y = ts_features, deg = 1)[0]
            self.pleth_features = np.concatenate((self.pleth_features, 
                                                  np.array([median, IQR, slope])), axis=0)
            self.pleth_feature_names += [ts_feature_names[i] + '_median', 
                                         ts_feature_names[i] + '_IQR', ts_feature_names[i] + '_slope']
    
    def find_r_peaks(self):
        """
        Andy's custom function to detect R-R interval:
        The default code works well if the ECG is not too noisy, but if it is, 
        you may want to play around with the different extration methods / tune parameters
        I made a custom function to do just this
        """
        arr2, _, _ = biosppy.tools.filter_signal(signal=self.ecg,
                                                 ftype='FIR',
                                                 band='bandpass',
                                                 order=int(0.3 * self.sampling_rate),
                                                 frequency=[3, 45],
                                                 sampling_rate=self.sampling_rate)

        rpeaks, = biosppy.signals.ecg.engzee_segmenter(arr2, 
                                                       sampling_rate=self.sampling_rate, 
                                                       threshold=self.threshold)
        rpeaks, = biosppy.signals.ecg.correct_rpeaks(signal=self.ecg,
                                                     rpeaks=rpeaks,
                                                     sampling_rate=self.sampling_rate,
                                                     tol=0.01)
    
        self.r_peaks = rpeaks
        self.filtered_ecg = arr2

    def get_nni(self):
        """
        Extract N-N intervals from ECG signals.
        """
        self.r_peaks_time = self.ecg_time[self.r_peaks]
        nni = pyhrv.tools.nn_intervals(self.r_peaks_time)
        nni = nni.astype(np.float32)
    
        rpeaks_median = np.median(self.filtered_ecg[self.r_peaks])
        idx = np.where(self.filtered_ecg[self.r_peaks] > self.rpeaks_ratio*rpeaks_median)[0]
    
        # remove the artifactual R peak and the NN interval between the artifactual R peak 
        # and its preceding and following peaks
        for i in idx:
            self.r_peaks[i] = -1
            self.r_peaks_time[i] = np.nan
            if i > 0:
                nni[i-1] = np.nan
            if i < len(nni):
                nni[i] = np.nan
        self.r_peaks = self.r_peaks[self.r_peaks != -1]
        self.r_peaks_time = self.r_peaks_time[~np.isnan(self.r_peaks_time)]
        nni = nni[~np.isnan(nni)]
    
        # remove those NN intervals which are abnormally long or short
        nni_median = np.median(nni)
        idx1 = np.where(nni > nni_median + self.nni_delta)[0]
        idx2 = np.where(nni < nni_median - self.nni_delta)[0]
    
        if len(idx1) > 0:
            nni[idx1] = np.nan
        if len(idx2) > 0:
            nni[idx2] = np.nan
        self.nni = nni[~np.isnan(nni)]
        
    def get_ecg_features(self):
        """
        Extract HR and HRV features from NN intervals.
        """
        # heart rates
        hr = pyhrv.tools.heart_rate(nni=self.nni)
        hr_median = np.median(hr)
        hr_iqr = np.quantile(hr, q=0.75) - np.quantile(hr, q=0.25)
        hr_slope = np.polyfit(x = np.arange(hr.shape[0]), y=hr, deg=1)[0]
    
        # time-domain features
        sdnn = pyhrv.time_domain.sdnn(nni=self.nni)[0]
        rmssd = pyhrv.time_domain.rmssd(nni=self.nni)[0]
    
        # frequency-domain features
        fft = pyhrv.frequency_domain.welch_psd(nni=self.nni, show=False, 
                                               show_param=False, legend=False)
        plt.clf()
        plt.close()
        vlf, lf, hf = fft['fft_abs']
        lf_norm, hf_norm = fft['fft_norm']
        lf_hf_ratio = fft['fft_ratio']
    
        # entropy
        sample_entropy = pyhrv.nonlinear.sample_entropy(nni=self.nni)[0]
        app_entropy = antropy.app_entropy(self.nni)
    
        features = np.array([hr_median, hr_iqr, hr_slope, sdnn, rmssd, vlf, lf, hf, 
                             lf_norm, hf_norm, lf_hf_ratio, sample_entropy, app_entropy])
        feature_names = ['ECG_HR_median', 'ECG_HR_IQR', 'ECG_HR_slope', 'ECG_SDNN', 'ECG_RMSSD', 
                         'ECG_VLF', 'ECG_LF', 'ECG_HF', 'ECG_LF_norm', 'ECG_HF_norm', 'ECG_LF_HF_ratio', 
                         'ECG_sample_entropy', 'ECG_approx_entropy']
    
        self.ecg_features = features
        self.ecg_feature_names = feature_names