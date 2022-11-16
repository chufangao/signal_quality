import numpy as np
import biosppy
import pyhrv
import antropy
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats

# def featurize_ecg_beat(beat_window):
#     ### beat_window is a 1D numpy array (can be of varying length)
#     pass

def featurize_ecg(window, sampling_rate, show=False):
    """Featurizes ECG into the following list of features. 
    Much thanks to Mononito Gowswami (https://github.com/mononitogoswami) for initial code.
    List of calculated features = ['ECG_HR_median', 'ECG_HR_IQR', 'ECG_HR_slope', 'ECG_SDNN', 'ECG_RMSSD',
        'ECG_VLF', 'ECG_LF', 'ECG_HF', 'ECG_LF_norm', 'ECG_HF_norm', 
        'ECG_LF_HF_ratio', 'ECG_sample_entropy', 'ECG_approx_entropy']

    Parameters
    ----------
    window : np.array
        1D numpy array of filtered ECG 
    sampling_rate : int
        The hz of the input ECG signal
    show : bool
        Whether to show the rpeak detection plots

    Returns
    ----------
    List : list
        List of 13 ECG features specified above
    """
    
    try:
        out = biosppy.signals.ecg.ecg(signal=window, sampling_rate=sampling_rate, show=show)
        nni = pyhrv.tools.nn_intervals(rpeaks=out['rpeaks'])
    
        hr = pyhrv.tools.heart_rate(nni=nni)
        hr_median = np.median(hr)
        hr_iqr = np.quantile(hr, q=0.75) - np.quantile(hr, q=0.25)
        hr_slope = np.polyfit(x = np.arange(hr.shape[0]), y=hr, deg=1)[0]

        # time-domain features
        sdnn = pyhrv.time_domain.sdnn(nni=nni)[0]
        rmssd = pyhrv.time_domain.rmssd(nni=nni)[0]

        # frequency-domain features
        fft = pyhrv.frequency_domain.welch_psd(nni=nni, show=False, show_param=False, legend=False)
        plt.clf()
        plt.close()

        vlf, lf, hf = fft['fft_abs']
        lf_norm, hf_norm = fft['fft_norm']
        lf_hf_ratio = fft['fft_ratio']

        # entropy
        sample_entropy = pyhrv.nonlinear.sample_entropy(nni=nni)[0]
        app_entropy = antropy.app_entropy(nni)

        return [hr_median, hr_iqr, hr_slope, 
            sdnn, rmssd, 
            vlf, lf, hf, 
            lf_norm, hf_norm, 
            lf_hf_ratio, 
            sample_entropy, app_entropy]
    except Exception as e:
        return [np.nan]*13

def featurize_pleth(window, pleth_time):
    """Featurizes Pleth into the following features. 
    Much thanks to Mononito Gowswami (https://github.com/mononitogoswami) for initial code.
    List of calculated variables: ['Pleth_systolic_amplitudes_median', 'Pleth_systolic_amplitudes_IQR', 'Pleth_systolic_amplitudes_slope', 
        'Pleth_peak_to_peak_interval_median', 'Pleth_peak_to_peak_interval_IQR', 'Pleth_peak_to_peak_interval_slope', 
        'Pleth_pulse_interval_median', 'Pleth_pulse_interval_IQR', 'Pleth_pulse_interval_slope', 
        'Pleth_upstroke_time_median', 'Pleth_upstroke_time_IQR', 'Pleth_upstroke_time_slope', 
        'Pleth_beat_skewness_median', 'Pleth_beat_skewness_IQR', 'Pleth_beat_skewness_slope']

    Parameters
    ----------
    window : np.array
        1D numpy array of filtered Pleth 
    pleth_time : np.array
        1D numpy array of timestamps that input window corresponds to

    Returns
    ----------
    List : list
        List of 15 Pleth features specified above    
    """
    ts_feature_names = ['Pleth_systolic_amplitudes', 'Pleth_peak_to_peak_interval', 'Pleth_pulse_interval',
        'Pleth_upstroke_time', 'Pleth_beat_skewness']

    # pleth_feature_names =  ['']*15

    try:
        # segment pleth ------------------------------------------------------
        beat_start, _ = scipy.signal.find_peaks(-1 * window, distance=50)
        if len(beat_start) > 0:

            beat_end = np.roll(beat_start, -1) # Roll start arrays by one before
            beat_start = beat_start[:-1]
            beat_end = beat_end[:-1]
            beat_start_time = pleth_time[beat_start]
            beat_end_time = pleth_time[beat_end]

            # find_systolic_peaks ------------------------------------------------------
            systolic_peaks, _ = scipy.signal.find_peaks(window[beat_start[0]:beat_end[-1]], distance=50) # The beats are at least 50 samples away. 
            systolic_peaks = systolic_peaks + beat_start[0]
            # Only compute beats from the start of the first beat to the end of the last beat
            # The amplitude is y-coordinate of the signal at the position of the peak
            systolic_peaks_time = pleth_time[systolic_peaks]
            systolic_amplitudes = window[systolic_peaks] 
            
            # find_upstroke_time ------------------------------------------------------
            try:
                upstroke_time = systolic_peaks_time - beat_start_time
            except: # Sometimes there is a mismatch between the number of systolic peaks and beats
                upstroke_time_ = []
                systolic_peaks_ = []
                systolic_amplitudes_ = []
                for i in range(len(beat_start)):
                    systolic_peak = systolic_peaks[systolic_peaks > beat_start[i]]
                    if len(systolic_peak) > 0: # Due to some issues, the last beat may not have an associated systolic peak
                        systolic_peak = systolic_peak[0] # Consider the closest systolic peak to the beat start time
                        amp_at_times = window[systolic_peak]
                        systolic_peaks_.append(systolic_peak)
                        systolic_amplitudes_.append(amp_at_times)
                        upstroke_time_.append(pleth_time[systolic_peak] - pleth_time[beat_start[i]])

                systolic_peaks = np.asarray(systolic_peaks_)
                systolic_peaks_time = pleth_time[systolic_peaks]
                systolic_amplitudes = np.asarray(systolic_amplitudes_)
                upstroke_time = np.asarray(upstroke_time_)

            # get_skewness_beats() ------------------------------------------------------
            beat_skewness = np.asarray([
                scipy.stats.skew(window[beat_start[i]:beat_end[i]]) 
                for i in range(beat_start.shape[0])
                ])

            # find_pulse_interval() ------------------------------------------------------
            pulse_interval = beat_start_time - np.roll(beat_start_time, 1)
            pulse_interval[0] = pulse_interval[1]

            # find_peak_to_peak_interval() ------------------------------------------------------
            peak_to_peak_interval = systolic_peaks_time - np.roll(systolic_peaks_time, 1)
            peak_to_peak_interval[0] = peak_to_peak_interval[1]

            # get_feature_aggregates() ------------------------------------------------------
            pleth_features = []
            # pleth_feature_names = []
            for i, ts_features in enumerate([systolic_amplitudes, peak_to_peak_interval, 
                                            pulse_interval, upstroke_time, beat_skewness]):
                median = np.median(ts_features)
                IQR = np.quantile(ts_features, q=0.75) - np.quantile(ts_features, q=0.25)
                slope = np.polyfit(x = np.arange(ts_features.shape[0]), y = ts_features, deg = 1)[0]
                pleth_features = pleth_features + [median, IQR, slope]
                # pleth_feature_names += [ts_feature_names[i] + '_median', ts_feature_names[i] + '_IQR', ts_feature_names[i] + '_slope']

        return pleth_features
    except Exception as e:
        return [np.nan]*15

