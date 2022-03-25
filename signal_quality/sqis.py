import numpy as np
import biosppy
import pyhrv
import matplotlib.pyplot as plt
import scipy.stats


def sqi_template_matching_correlation(ecg_window, sampling_rate, show=False):
    """
    C. Orphanidou, T. Bonnici, P. Charlton, D. Clifton, D. Vallance and L. Tarassenko, 
    "Signal quality indices for the electrocardiogram and photoplethysmogram: Derivation and applications to wireless monitoring", 
    IEEE J. Biomed. Health Informat., vol. 19, no. 3, pp. 832-838, May 2015.

    ecg_window = input ECG as a 1d numpy array
    sampling_rate = the hz of the input ECG signal
    show = a boolean value whether to show the obtained peaks

    returns average correlation coefficient (scipy.stats.pearsonr)
        that range from -1 to 1
    """

    
    ## names = ('ts', 'filtered', 'rpeaks', 'templates_ts', 'templates',
    #          'heart_rate_ts', 'heart_rate')

    out = biosppy.signals.ecg.ecg(signal=ecg_window, sampling_rate=sampling_rate, show=show)
    nni = pyhrv.tools.nn_intervals(rpeaks=out['rpeaks'])
    # nni is in ms, convert to s
    nni = nni / 1000

    # obtain median rr interval
    median_qrs_window = np.median(out['rpeaks'][1:] - out['rpeaks'][:-1]).astype(int)

    # check heart rate in reasonable range of [40,180]
    if np.any(out['heart_rate'] < 40) or np.any(180 < out['heart_rate']):
        return 1

    # if all nni are less than 3 seconds
    if np.any(nni > 3):
        return 1

    # check max_rr_interval / min_rr_interval < 2.2
    if (np.max(nni) / np.min(nni)) > 2.2:
        return 1

    templates = np.array([
        ecg_window[r_peak-median_qrs_window//2:r_peak+median_qrs_window//2] 
        for r_peak in out['rpeaks']
        if (r_peak-median_qrs_window//2 >= 0) and (r_peak+median_qrs_window//2 < len(ecg_window))
    ])
    
    average_template = np.mean(templates, axis=0)

    # scipy.stats.pearsonr returns r, p_value
    corrcoefs = [
        scipy.stats.pearsonr(x=templates[i], y=average_template)[0]
        for i in range(len(templates))
        ]

    return np.mean(corrcoefs)
