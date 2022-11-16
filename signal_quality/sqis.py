import numpy as np
import biosppy
import pyhrv
import matplotlib.pyplot as plt
import scipy.stats
import neurokit2 as nk
import antropy
import sklearn

### ===================================================== Relevant ECG Features =====================================================

def orphanidou2015_sqi(ecg_cleaned, sampling_rate, show=False):
    """ Implementation of template matching approach introduced by Orphanidou et al. 
    Returns the average correlation coefficient (scipy.stats.pearsonr) of the QRS waveforms that ranges from -1 to 1

    Parameters
    ----------
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    sampling_rate : int
        The hz of the input ECG signal
    show : bool
         Flag whether to show the obtained peaks

    Reference
    ----------
    C. Orphanidou, T. Bonnici, P. Charlton, D. Clifton, D. Vallance and L. Tarassenko, 
    "Signal quality indices for the electrocardiogram and photoplethysmogram: Derivation and applications to wireless monitoring", 
    IEEE J. Biomed. Health Informat., vol. 19, no. 3, pp. 832-838, May 2015.
    """
    
    try:
        ## out = ('ts', 'filtered', 'rpeaks', 'templates_ts', 'templates', 'heart_rate_ts', 'heart_rate')
        out = biosppy.signals.ecg.ecg(signal=ecg_cleaned, sampling_rate=sampling_rate, show=show)
    except Exception as e:
        return np.nan

    nni = pyhrv.tools.nn_intervals(rpeaks=out['rpeaks'])
    ## nni is in ms, convert to s
    nni = nni / 1000

    ## obtain median rr interval
    median_qrs_window = np.median(out['rpeaks'][1:] - out['rpeaks'][:-1]).astype(int)

    ## check heart rate in reasonable range of [40,180]
    if np.any(out['heart_rate'] < 40) or np.any(180 < out['heart_rate']):
        return 1.

    ## if all nni are less than 3 seconds
    if np.any(nni > 3):
        return 1.

    ## check max_rr_interval / min_rr_interval < 2.2
    if (np.max(nni) / np.min(nni)) > 2.2:
        return 1.

    templates = np.array([
        ecg_cleaned[r_peak-median_qrs_window//2:r_peak+median_qrs_window//2] 
        for r_peak in out['rpeaks']
        if (r_peak-median_qrs_window//2 >= 0) and (r_peak+median_qrs_window//2 < len(ecg_cleaned))
    ])
    
    average_template = np.mean(templates, axis=0)

    ## scipy.stats.pearsonr returns r, p_value
    corrcoefs = [
        scipy.stats.pearsonr(x=templates[i], y=average_template)[0]
        for i in range(len(templates))
        ]

    return np.mean(corrcoefs)

def averageQRS_sqi(ecg_cleaned, sampling_rate):
    """Computes a continuous index of quality of the ECG signal by interpolating the distance
    of each QRS segment from the average QRS segment present in the data. This index is relative: 
    1 corresponds to heartbeats that are the closest to the average sample and 0 corresponds to 
    the most distant heartbeat from that average sample. Note that 1 does not necessarily means
    "good": if the majority of samples are bad, than being close to the average will likely mean 
    bad as well. Use this index with care and plot it alongside your ECG signal to see if it makes sense.

    Parameters
    ----------
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    sampling_rate : int
        The hz of the input ECG signal

    Reference
    ----------
    Source: https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/ecg/ecg_quality.py
    """
    try:
        rating = nk.ecg_quality(ecg_cleaned=ecg_cleaned, rpeaks=None, sampling_rate=sampling_rate, method="averageQRS")

        if rating == "Excellent":
            return 2
        elif rating == "Unnacceptable":
            return 0
        else:
            return 1

    except Exception as e:
        # print(e)
        return np.nan

def zhao2018_sqi(ecg_cleaned, sampling_rate):
    """Returns the zhao2018 output for ECG signal quality prediction. 
    The metod extracts several signal quality indexes (sqis):
    QRS wave power spectrum distribution psqi, kurtosis ksqi, and baseline relative power bassqi.
    An additional R peak detection match qsqi was originally computed in the paper but left out
    in this algorithm. The indices were originally weighted with a ratio of [0.4, 0.4, 0.1, 0.1] to
    generate the final classification outcome, but because qsqi was dropped,
    the weights have been rearranged to [0.6, 0.2, 0.2] for psqi, ksqi and bassqi respectively

    Parameters
    ----------
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).

    Reference
    ----------
    Source: https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/ecg/ecg_quality.py
    """
    try:
        rating = nk.ecg_quality(ecg_cleaned=ecg_cleaned, rpeaks=None, sampling_rate=sampling_rate, method="zhao2018", approach='fuzzy')
        if rating == "Excellent":
            return 2
        elif rating == "Unnacceptable":
            return 0
        else:
            return 1
    except Exception as e:
        # print(e)
        return np.nan


def p_sqi(ecg_cleaned, sampling_rate, window, num_spectrum=[5, 15], dem_spectrum=[5, 40]):
    """Returns a sqi based off of the Power Spectrum Distribution of QRS Waveform.

    Parameters
    ----------
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    window : int
        Length of each window in seconds. See `signal_psd()`.

    Reference
    ----------
    Source: https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/ecg/ecg_quality.py
    """
    try:
        psd = nk.signal_power(
            ecg_cleaned,
            sampling_rate=sampling_rate,
            frequency_band=[num_spectrum, dem_spectrum],
            method="welch",
            normalize=False,
            window=window
            )

        num_power = psd.iloc[0][0]
        dem_power = psd.iloc[0][1]

        return num_power / dem_power
    except Exception as e:
        return np.nan


def bas_sqi(ecg_cleaned, sampling_rate, window, num_spectrum=[0, 1], dem_spectrum=[0, 40]):
    """Returns a sqi that measures Relative Power in the baseline.

    Parameters
    ----------
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    window : int
        Length of each window in seconds. See `signal_psd()`.

    Reference
    ----------
    Source: https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/ecg/ecg_quality.py
    """

    try:
        psd = nk.signal_power(
            ecg_cleaned,
            sampling_rate=sampling_rate,
            frequency_band=[num_spectrum, dem_spectrum],
            method="welch",
            normalize=False,
            window=window
            )

        num_power = psd.iloc[0][0]
        dem_power = psd.iloc[0][1]

        return 1 - num_power / dem_power
    except Exception as e:
        return np.nan

def c_sqi(ecg_cleaned, sampling_rate):
    """Returns a sqi that measures variability in the R-R Interval. 
    When an artifact is present, the QRS detector underperforms by either
    missing R-peaks or erroneously identifying noisy peaks as Rpeaks. The
    above two problems will lead to a high degree of variability in the
    distribution of R-R intervals.

    Parameters
    ----------
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    sampling_frequency : int
        Input ecg sampling frequency

    Reference
    ----------
    Source: https://github.com/Aura-healthcare/ecg_qc/blob/main/ecg_qc/sqi_computing/sqi_rr_intervals.py
    """
    try:
        rri_list = biosppy.signals.ecg.hamilton_segmenter(signal=ecg_cleaned, sampling_rate=sampling_rate)[0]
        c_sqi_score = np.std(rri_list, ddof=1) / np.mean(rri_list)

    except Exception:
        c_sqi_score = np.nan

    return c_sqi_score

def _get_num_matching(qrs_frames_1, qrs_frames_2, frame_tolerance):
    """Helper function to determine number of matching Rpeaks 
    """
    ## Catch complete failed QRS detection
    if (len(qrs_frames_1) == 0 or len(qrs_frames_2) == 0):
        return 0

    i = 0
    j = 0
    matching_frames = 0

    while i < len(qrs_frames_1) and j < len(qrs_frames_2):
        min_qrs_frame = min(qrs_frames_1[i], qrs_frames_2[j])
        ## Get missing detected beats intervals
        ## Matching frames
        if abs(qrs_frames_2[j] - qrs_frames_1[i]) < frame_tolerance:
            matching_frames += 1
            i += 1
            j += 1
        else:
            ## increment first QRS in frame list
            if min_qrs_frame == qrs_frames_1[i]:
                i += 1
            else:
                j += 1
    return matching_frames

def q_sqi(ecg_cleaned, sampling_rate, matching_qrs_frames_tolerance=50, method='q_sqi'):
    """Returns a sqi that measures matching Degree of R Peak Detection. 
    Two R wave detection algorithms are compared with their respective number
    of R waves detected (Hamilton vs Stationary Wavelet Transform).
    
    Parameters
    ----------
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    sampling_frequency : int
        Input ecg sampling frequency

    Reference
    ----------
    Source: https://github.com/Aura-healthcare/ecg_qc/blob/main/ecg_qc/sqi_computing/sqi_rr_intervals.py
    """

    ## returns signals: df, info: dict of {'ECG_R_Peaks', 'sampling_rate'}
    if method=='q_sqi':
        qrs_frames_1 = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='hamilton2002')[1]['ECG_R_Peaks']
        qrs_frames_2 = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='kalidas2017')[1]['ECG_R_Peaks']
        ## compute_qrs_frames_correlation
        # single_frame_duration = 1/sampling_rate
        frame_tolerance = matching_qrs_frames_tolerance * 0.001 * sampling_rate
    elif method=='b_sqi':
        qrs_frames_1 = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='zong2003')[1]['ECG_R_Peaks']
        qrs_frames_2 = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='pantompkins1985')[1]['ECG_R_Peaks']
        ## compute_qrs_frames_correlation (150 ms)
        frame_tolerance = .15*sampling_rate
    else:
        raise NotImplementedError

    ## Catch complete failed QRS detection

    matching_frames = _get_num_matching(qrs_frames_1, qrs_frames_2, frame_tolerance)

    if method=='q_sqi':
        if (len(qrs_frames_1) == 0 or len(qrs_frames_2) == 0): # error
            return 0.
        else:
            return 2 * matching_frames / (len(qrs_frames_1) + len(qrs_frames_2))
    elif method=='b_sqi':
        if (len(qrs_frames_1) + len(qrs_frames_2) - matching_frames) == 0: # error
            return 0.
        else:
            return matching_frames / (len(qrs_frames_1) + len(qrs_frames_2) - matching_frames)

def i_sqi(ecg_cleaned_list, sampling_rate):
    """Returns a sqi that measures inter-channel signal quality. 
    Calculated as the ratio of the number of matched beats (Nmatched) to 
    all detected beats (Nall) between a given lead and all other synchronous ECG.

    Parameters
    ----------
    ecg_cleaned_list : List of np.array
        List of cleaned ECG signal in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).

    Reference
    ----------
    Li, Qiao, Roger G. Mark, and Gari D. Clifford. 
    "Robust heart rate estimation from multiple asynchronous noisy sources using signal quality indices and a Kalman filter." 
    Physiological measurement 29.1 (2007): 15.    
    """
    qrs_frames_list = [nk.ecg_peaks(ecg_cleaned, sampling_rate, method='hamilton2002')[1]['ECG_R_Peaks'] for ecg_cleaned in ecg_cleaned_list]
    frame_tolerance = .15*sampling_rate
    
    max_matching = 0
    for i in range(len(qrs_frames_list)):
        for j in range(i+1, len(qrs_frames_list)):
            matching = _get_num_matching(qrs_frames_list[i], qrs_frames_list[j], frame_tolerance)
            max_matching = max(max_matching, matching)

    return max_matching

def bs_sqi(ecg_cleaned, peaks, sampling_rate):
    """Returns a sqi for baseline wander check in time domain. The higher the wander, the lower the bs_sqi.

    Parameters
    ----------
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    peaks : list 
        List of rpeak locations like in nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='kalidas2017')[1]['ECG_R_Peaks']
    sampling_frequency : int
        Input ecg sampling frequency

    Reference
    ----------
    Source: Li, Qiao, Cadathur Rajagopalan, and Gari D. Clifford. 
    "A machine learning approach to multi-level ECG signal quality classification." 
    Computer methods and programs in biomedicine 117.3 (2014): 435-447. 
    """

    # peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='kalidas2017')[1]['ECG_R_Peaks']
    filtered = nk.signal_filter(signal=ecg_cleaned, sampling_rate=sampling_rate, highcut=1, method='butterworth')
    
    total = []
    
    for peak in peaks:
        # Ra is peak to peak amplitude of ECG signal from (R-0.07s, R+0.08s)
        window = ecg_cleaned[max(int(peak-0.07*sampling_rate), 0) : min(int(peak+0.08*sampling_rate), len(ecg_cleaned))]
        Ra = np.max(window) - np.min(window)

        # Ba is peak to peak amplitude of baseline (1hz lowpass filter) from (R-1s, R+1s)
        window = filtered[max(int(peak-1*sampling_rate), 0) : min(int(peak+1*sampling_rate), len(ecg_cleaned))]
        Ba = np.max(window) - np.min(window)

        total.append(Ra / Ba)
    
    total = np.nanmean(total)
    return total

def e_sqi(ecg_cleaned, peaks, sampling_rate):
    """ Returns a sqi based off of the sum of energy of detected QRS waveforms over energy of entire signal.

    Parameters
    ----------
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    peaks : list 
        List of rpeak locations like in nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='kalidas2017')[1]['ECG_R_Peaks']
    sampling_frequency : int
        Input ecg sampling frequency

    Reference
    ----------
    Source: Li, Qiao, Cadathur Rajagopalan, and Gari D. Clifford. 
    "A machine learning approach to multi-level ECG signal quality classification." 
    Computer methods and programs in biomedicine 117.3 (2014): 435-447. 
    """
    total = 0
    
    for peak in peaks:
        # Ra is peak to peak amplitude of ECG signal from (R-0.07s, R+0.08s)
        window = ecg_cleaned[max(int(peak-0.07*sampling_rate), 0) : min(int(peak+0.08*sampling_rate), len(ecg_cleaned))]    
        # energy of QRS
        total += np.dot(window, window)

    return total / np.dot(ecg_cleaned, ecg_cleaned)

def hf_sqi(ecg_raw, peaks, sampling_rate):
    """ Returns a sqi based on the relative amplitude of high frequency noise.

    Parameters
    ----------
    ecg_raw : np.array
        Input unfiltered ECG signal
    peaks : list 
        List of rpeak locations like in nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='kalidas2017')[1]['ECG_R_Peaks']
    sampling_frequency : int
        Input ecg sampling frequency

    Reference
    ----------
    Source: Li, Qiao, Cadathur Rajagopalan, and Gari D. Clifford. 
    "A machine learning approach to multi-level ECG signal quality classification." 
    Computer methods and programs in biomedicine 117.3 (2014): 435-447. 
    """
    if len(ecg_raw) < 6: return np.nan

    ## integer coefficients high pass filter y(j) = x(j) − 2x(j − 1) + x(j − 2)
    y = ecg_raw[:-2] - 2*ecg_raw[1:-1] + ecg_raw[2:]
    s = y[:-5] + y[1:-4] + y[2:-3] + y[3:-2] + y[4:-1] + y[5:]
    
    total = []
    for peak in peaks:
        # Ra is peak to peak amplitude of ECG signal from (R-0.07s, R+0.08s)
        window = ecg_raw[max(int(peak-0.07*sampling_rate), 0) : min(int(peak+0.08*sampling_rate), len(ecg_raw))]    
        # energy of QRS

        Ra = np.max(window) - np.min(window)

        window = s[max(int(peak-0.28*sampling_rate), 0) : min(int(peak-0.05*sampling_rate), len(s))]
        H = np.nanmean(window)

        total.append(Ra / H)
    return np.nanmean(total)

def rsd_sqi(ecg_cleaned, peaks, sampling_rate):
    """ Returns a sqi based on the relative standard deviation.

    Parameters
    ----------
    ecg_raw : np.array
        Input unfiltered ECG signal
    peaks : list 
        List of rpeak locations like in nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='kalidas2017')[1]['ECG_R_Peaks']
    sampling_frequency : int
        Input ecg sampling frequency

    Reference
    ----------
    Source: Li, Qiao, Cadathur Rajagopalan, and Gari D. Clifford. 
    "A machine learning approach to multi-level ECG signal quality classification." 
    Computer methods and programs in biomedicine 117.3 (2014): 435-447. 
    """
    
    total = []
    for peak in peaks:
        # Ra is peak to peak amplitude of ECG signal from (R-0.07s, R+0.08s)
        window = ecg_cleaned[max(int(peak-0.07*sampling_rate), 0) : min(int(peak+0.08*sampling_rate), len(ecg_cleaned))]    
        # energy of QRS
        sigma_r = np.nanstd(window)

        window = ecg_cleaned[max(int(peak-0.2*sampling_rate), 0) : min(int(peak+0.2*sampling_rate), len(ecg_cleaned))]
        sigma_a = np.nanstd(window)
        
        total.append(sigma_r / (sigma_a*2))
    return np.nanmean(total)

def get_ecg_sqis(ecg_raw, ecg_cleaned, peaks, sampling_rate, window):
    """ Returns all ECG sqis.

    Parameters
    ----------
    ecg_raw : np.array
        Input unfiltered ECG signal
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    peaks : list 
        List of rpeak locations like in nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='kalidas2017')[1]['ECG_R_Peaks']
    sampling_frequency : int
        Input ecg sampling frequency
    window : int
        Length of each window in seconds. See `signal_psd()`.

    Returns
    ----------
    List : list
        List of 11 implemented single channel ECG sqis and 10 time series sqis
        for a total of 22 features 
    """
    # ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=sampling_rate, method="neurokit")
    # peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='kalidas2017')[1]['ECG_R_Peaks']

    ecg_sqis = [
        orphanidou2015_sqi(ecg_cleaned, sampling_rate, show=False),
        averageQRS_sqi(ecg_cleaned, sampling_rate),
        zhao2018_sqi(ecg_cleaned, sampling_rate),
        p_sqi(ecg_cleaned, sampling_rate, window, num_spectrum=[5, 15], dem_spectrum=[5, 40]),
        bas_sqi(ecg_cleaned, sampling_rate, window, num_spectrum=[0, 1], dem_spectrum=[0, 40]),
        c_sqi(ecg_cleaned, sampling_rate),
        q_sqi(ecg_cleaned, sampling_rate, matching_qrs_frames_tolerance=50),
        q_sqi(ecg_cleaned, sampling_rate, method='b_sqi'),
        bs_sqi(ecg_cleaned, peaks, sampling_rate),
        e_sqi(ecg_cleaned, peaks, sampling_rate),
        hf_sqi(ecg_raw, peaks, sampling_rate),
        rsd_sqi(ecg_cleaned, peaks, sampling_rate)
    ]

    generic_sqis = get_generic_sqis(signal=ecg_raw)
    return ecg_sqis + generic_sqis

### ===================================================== Relevant Pleth Features =====================================================

def perfusion_sqi(pleth_raw, pleth_cleaned):
    """Returns perfusion of Pleth. The perfusion index is the ratio of the pulsatile blood flow to the nonpulsatile 
    or static blood in peripheral tissue. In other words, it is the difference of the 
    amount of light absorbed through the pulse of when light is transmitted through 
    the finger. It is calculated as AC/DC * 100

    Parameters
    ----------
    pleth_raw : np.array
        Input unfiltered Pleth signal
    pleth_cleaned : np.array
        Input filtered Pleth signal
    """
    try:
        return (np.nanmax(pleth_cleaned) - np.nanmin(pleth_cleaned)) / np.nanmean(pleth_raw) * 100
    except Exception as e:
        return np.nan    

def get_pleth_sqis(pleth_raw, pleth_cleaned):
    """ Returns all Pleth sqis.

    Parameters
    ----------
    pleth_raw : np.array
        Input unfiltered Pleth signal
    pleth_cleaned : np.array
        Input filtered Pleth signal
    
    Returns
    ----------
    List : list
        List of 1 implemented single channel Pleth sqi and 10 time series sqis
        for a total of 11 features 
    """
    # pleth_cleaned = nk.ppg_clean(ppg_signal=pleth_raw, sampling_rate=sampling_rate, method='elgendi')
    pleth_sqis = [
        perfusion_sqi(pleth_raw, pleth_cleaned),
    ]

    generic_sqis = get_generic_sqis(signal=pleth_raw)
    return pleth_sqis + generic_sqis

### ===================================================== Relevant General Time Series Features =====================================================

def k_sqi(signal, kurtosis_method='fisher'):
    """Return the kurtosis of the signal, with Fisher's or Pearson's method.

    Parameters
    ----------
    signal : np.array
        The input signal
    kurtosis_method : str
        Compute kurtosis (ksqi) based on "fisher" (default) or "pearson" definition.

    Reference
    ----------
    Source: https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/ecg/ecg_quality.py
    """
    if kurtosis_method == "fisher":
        return scipy.stats.kurtosis(signal, fisher=True)
    elif kurtosis_method == "pearson":
        return scipy.stats.kurtosis(signal, fisher=False)

def s_sqi(signal):
    """Return the skewness of the signal.

    Parameters
    ----------
    signal : np.array
        The input signal
    """
    return scipy.stats.skew(signal)

def pur_sqi(signal):
    """ Returns the signal purity of the input. 
    In the case of a periodic signal with a single dominant frequency, 
    it takes the value of one and approaches zero for non-sinusoidal noisy signals.
    antropy.hjorth_params returns 2 floats: (mobility, complexity).
    Complexity is the value we want.

    Parameters
    ----------
    signal : np.array
        The input signal
    """
    return antropy.hjorth_params(signal)[1]

def ent_sqi(signal):
    """ Returns the sample entropy of the signal.

    Parameters
    ----------
    signal : np.array
        The input signal
    """
    return antropy.sample_entropy(signal)

def pca_sqi(signal):
    """ Returns a PCA sqi of the input signals.

    Parameters 
    ----------
    signal : array-like of shape (n_samples, n_features)
        Multivariate time-series, shape is at least 2 dimensional
    """
    # todo: Currently, we are only using single channel (1d) swis
    pca = sklearn.decomposition.PCA(n_components=None)
    pca.fit(signal)

    return np.sum(pca.singular_values_[:5]) / np.sum(pca.singular_values_)

def autocorr_sqi(signal, lag):
    """Calculates the autocorrelation of the specified lag, according to the formula in https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    Parameters
    ----------
    signal : np.array
        The input signal
    lag : int
        The lag to use forthe autocorrelation calculation of the signal

    Reference
    ----------    
    source: https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#autocorrelation
    """
    # This is important: If a series is passed, the product below is calculated
    # based on the index, which corresponds to squaring the series.

    if len(signal) < lag:
        return np.nan
    # Slice the relevant subseries based on the lag
    y1 = signal[: (len(signal) - lag)]
    y2 = signal[lag:]
    # Subtract the mean of the whole series x
    x_mean = np.mean(signal)
    # The result is sometimes referred to as "covariation"
    sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
    # Return the normalized unbiased covariance
    v = np.var(signal)
    if np.isclose(v, 0):
        return np.nan
    else:
        return sum_product / ((len(signal) - lag) * v)

def zc_sqi(signal):
    """Returns the zero crossing rate.

    Parameters
    ----------
    signal : np.array
        The input signal
    """
    return antropy.num_zerocross(signal)

def snr_sqi(signal_raw, signal_cleaned):
    """Returns the signal to noise ratio (SNR). There are many ways to define SNR, here, we use std of filtered vs std of raw signal.

    Parameters
    ----------
    signal_raw : np.array
        Raw input signal
    signal_cleaned : np.array    
        Cleaned input signal
    """
    return np.std(np.abs(signal_cleaned)) / np.std(np.abs(signal_raw))

def f_sqi(signal, window_size=3, threshold=1e-7):
    """Returns an sqi that computes percentage of flatness in the signal. 
    Constant values over a longer period (flat line) may be caused by sensor failures.

    Parameters
    ----------
    signal : np.array
        The input signal
    window_size : int
        Window to detect flat line, larger values will lower detection sensitivity
    threshold : float
        Threshold of flatness. I.e. Where (max-min) is considered equivalent  

    Reference
    ----------
    Source: https://github.com/DHI/tsod/blob/main/tsod/detectors.py
    """
    if window_size >= len(signal): return 0

    rolling = np.lib.stride_tricks.sliding_window_view(signal, window_shape=window_size)
    rollmax = np.nanmax(rolling, axis=1)
    rollmin = np.nanmin(rolling, axis=1)
    
    anomalies = rollmax - rollmin < threshold
    anomalies[0] = False  # first element cannot be determined
    anomalies[-1] = False

    idx = np.where(anomalies)[0]
    if idx is not None:
        # assuming window size = 3
        # remove also points before and after each detected anomaly
        anomalies[idx[idx > 0] - 1] = True
        maxidx = len(anomalies) - 1
        anomalies[idx[idx < maxidx] + 1] = True

    return np.sum(anomalies) / len(anomalies)

def get_generic_sqis(signal):
    """ Returns sqis for a generic signal.

    Parameters
    ----------
    signal : np.array
        The input signal

    Returns
    ----------
    List : list
        List of 10 time series sqis
    """
    return [
        k_sqi(signal, kurtosis_method='fisher'),
        s_sqi(signal),
        pur_sqi(signal),
        ent_sqi(signal),
        zc_sqi(signal),
        f_sqi(signal, window_size=3, threshold=1e-7),
        np.nanmean(signal),
        np.nanstd(signal),
        np.nanmax(signal),
        np.nanmin(signal)
    ]