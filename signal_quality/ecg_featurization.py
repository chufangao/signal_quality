import numpy as np
import biosppy
import pyhrv
import antropy
import matplotlib.pyplot as plt

def featurize_beats(beats):
    ### beats is list of 1D numpy arrays (can be of varying length)
    pass


def featurize_windows(windows, sampling_rate, show=False):
    ### windows is list of 1D numpy arrays (can be of varying length)
    feature_names = ['ECG_HR_median', 'ECG_HR_IQR', 'ECG_HR_slope', 'ECG_SDNN', 'ECG_RMSSD', 
                        'ECG_VLF', 'ECG_LF', 'ECG_HF', 'ECG_LF_norm', 'ECG_HF_norm', 'ECG_LF_HF_ratio', 
                        'ECG_sample_entropy', 'ECG_approx_entropy']
    features = []
    
    for window in windows:
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
        fft = pyhrv.frequency_domain.welch_psd(nni=nni, show=False, 
                                               show_param=False, legend=False)
        plt.clf()
        plt.close()

        vlf, lf, hf = fft['fft_abs']
        lf_norm, hf_norm = fft['fft_norm']
        lf_hf_ratio = fft['fft_ratio']
    
        # entropy
        sample_entropy = pyhrv.nonlinear.sample_entropy(nni=nni)[0]
        app_entropy = antropy.app_entropy(nni)

        features.append(np.array([
            hr_median, hr_iqr, hr_slope, 
            sdnn, rmssd, 
            vlf, lf, hf, 
            lf_norm, hf_norm, 
            lf_hf_ratio, 
            sample_entropy, app_entropy]))
    
    return features