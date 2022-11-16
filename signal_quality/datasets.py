import numpy as np
import wfdb
import matplotlib.pyplot as plt
import biosppy 
import scipy
import sklearn
from tqdm import tqdm

def load_ecg_by_windows(channel, rates, noise_threshold, window_size=60*125, step=30*125):
    """
    channel = the entire raw ECG waveform of shape (length,)
    rates = the processed (numeric) annotations of same shape as ECG (length,)
    sampling_rate = the hz of the input ECG signal
    downsampled_sampling_rate = the hz to downsample to
    window_size = the window, after downsampling, of indices in a feature window
    """
    
    # Split into individual heartbeats. For each heartbeat
    # record, append classification (normal/abnormal).
    labels = []
    beats = []

    # Skip first and last beat.
    for idxval in range(window_size, len(channel), step):
        # Get the classification value that is on
        # or near the position of the rpeak index.
        # set as abnormal beat if it exists

        if rates is not None:
            window_labels = rates[max(0,idxval-window_size):idxval]
            
            # Skip beat if there is no classification.
            if (window_labels==-1.0).sum() == len(window_labels):
                continue

            # print((window_labels==1.0).sum() / len(window_labels))
            if ((window_labels==1.0).sum() / len(window_labels)) > noise_threshold:
                catval = 1
            else:
                catval = 0
            labels.append(catval)

        # Append some extra readings around the beat.
        beat = channel[max(0,idxval-window_size):idxval]

        # # Normalize the readings to a 0-1 range for ML purposes.
        # beat_range = beat.max() - beat.min()
        # if beat_range == 0:
        #     continue
        # beat = (beat - beat.min()) / beat_range

        beats.append(beat)

    # return data and labels
    return beats, labels

def load_ecg_beat_by_beat(channel, rates, sampling_rate, beat_window=90, show=False):
    """
    channel = the entire raw ECG waveform of shape (length,)
    rates = the processed (numeric) annotations of same shape as ECG (length,)
    sampling_rate = the hz of the input ECG signal
    downsampled_sampling_rate = the hz to downsample to
    beat_window = the window, after downsampling, of indices to get around the peak
    show = a boolean value whether to show the obtained peaks
    """
    # Instead of using the annotations to find the beats, we will
    # use R-peak detection instead. The reason for this is so that
    # the same logic can be used to analyze new and un-annotated
    # ECG data. We use the annotations here only to classify the
    # beat as either Normal or Abnormal and to train the model.

    # Find rpeaks in the ECG data. Most should match with
    # the annotations.
    # biosppy.signals.ecg.ecg returns:
    # names = ('ts', 'filtered', 'rpeaks', 'templates_ts', 'templates',
    #          'heart_rate_ts', 'heart_rate')
    out = biosppy.signals.ecg.ecg(signal=channel, sampling_rate=sampling_rate, show=show)
    
    # Split into individual heartbeats. For each heartbeat
    # record, append classification (normal/abnormal).
    labels = []
    beats = []

    # Skip first and last beat.
    for idxval in out['rpeaks'][1:-1]:
        # Get the classification value that is on
        # or near the position of the rpeak index.
        # set as abnormal beat if it exists

        if rates is not None:
            catval = rates[max(0,idxval-beat_window):idxval+beat_window].max()
        
            # Skip beat if there is no classification.
            if catval == -1.0:
                continue            
            labels.append(catval)

        # Append some extra readings around the beat.
        beat = channel[max(0,idxval-beat_window):idxval+beat_window]

        if show:
            plt.plot(beat)
            plt.scatter(beat_window, 1, c='r', label='detected peak')
            plt.legend(loc='upper right'); plt.grid(); plt.ylabel('ECG'); plt.xlabel(str(sampling_rate)+' hz'); plt.show()

        # # Normalize the readings to a 0-1 range for ML purposes.
        # beat_range = beat.max() - beat.min()
        # if beat_range == 0:
        #     continue
        # beat = (beat - beat.min()) / beat_range

        beats.append(beat)

    # return data and labels
    return beats, labels

def get_power(signal):
    """" Helper function to return average power of signal
    """
    return np.mean(np.power(signal, 2))

def calc_snr(signal, noise):
    """" Calculates signal to noise ratio
    """
    signal_avg_watts = get_power(signal)
    noise_avg_watts = get_power(noise)
    return 10*(np.log10(signal_avg_watts) - np.log10(noise_avg_watts))


def add_noise(signal, method, target_snr_db=18, random_state=0, verbose=False):
    """ Adds noise of specified target_snr_db dB to signal

    method = type of noise to add
    """

    np.random.seed(random_state)
    # Adding noise using target SNR

    # Calculate signal power and convert to dB 
    sig_avg_watts = get_power(signal)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)

    if method == 'gaussian':
        # Generate an sample of white noise
        noise_volts = np.random.normal(0, np.sqrt(noise_avg_watts), len(signal))
    else:
        raise NotImplementedError

    if verbose:
        print('SNR', calc_snr(signal, noise_volts))

    # Noise up the original signal
    return signal + noise_volts


def add_noisy_signal(signal, noise, target_snr_db=18, verbose=False):
    """ Adds noise of specified target_snr_db dB to signal via rescaling
    """
    # Calculate signal power and convert to dB 
    sig_avg_watts = get_power(signal)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)

    # rescale noise
    noise -= np.mean(noise)
    noise *= np.sqrt(noise_avg_watts) / noise.std()

    if verbose:
        print('SNR', calc_snr(signal, noise))

    # Noise up the original signal
    return signal + noise


def load_nstdb_extra(nstdb_path, mitdb_path, verbose=False, downsampled_sampling_rate=125):
    """ 
    https://physionet.org/content/nstdb/1.0.0/

    ## 1. Install the WFDB package: https://www.physionet.org/content/wfdb/
    ##      Source: Moody, G., Pollard, T., & Moody, B. (2021). WFDB Software Package (version 10.6.2). PhysioNet. https://doi.org/10.13026/zzpx-h016.
    ##      Source: Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: 
    ##          Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
    ##
    ## 2. Download the MIT-BIH Noise Stress Test Database: https://physionet.org/content/nstdb/ and rename it to "nstdb"
    ##      Source: Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia detectors. Computers in Cardiology 1984; 11:381-384.
    ##
    ## 3. Download the MIT-BIH: https://physionet.org/content/mitdb/ and rename it to "mitdb"
    ##      Source: Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
    ##
    ## 4. Run "bash generate_nstdb_extra.sh" to create the nstdb_extra directory
    ##      This takes in 2 clean ECGs (118 and 119) and adds 3 types of noise (baseline wander, muscle EMG artifact, electrode motion artifact)
    ##      at different signal-to-noise ratios (24,18,12,6,0,-6) in dB. Original script only adds one type of noise, this script adds all 3 types.
    ##
    ##      Source: https://physionet.org/content/nstdb/1.0.0/nstdbgen
    ##      Source: https://physionet.org/physiotools/wag/nst-1.htm
    ##

    data_path = path where the .dat, .hea, .atr files are located for every patient
    """
    subjects = ['118', '119']

    output_dict = {}
    for subject in subjects:
        # Read in the data
        record = wfdb.rdrecord(mitdb_path+subject)
        data = record.p_signal.transpose()

        for noise_type in ['em', 'ma', 'bw', 'gn']:                    
            for dB in ['_6','00','06','12','18','24']:

                int_db = int(dB.replace('_','-'))
                subject_str = subject+noise_type+dB
                output_dict[subject_str] = {}

                # Process each channel separately (2 per input file).
                for channelid, channel in enumerate(data):
                    chname = record.sig_name[channelid]
                    # Resample from 360Hz to 125Hz
                    newsize = int((len(channel) * downsampled_sampling_rate / record.fs) + 0.5)
                    channel = scipy.signal.resample(channel, newsize)

                            
                    if verbose:
                        print('Name',subject_str,'ECG channel type:',chname)
                    
                    if noise_type in ['em', 'ma', 'bw']:
                        noise_record = wfdb.rdrecord(nstdb_path+noise_type)
                        noise_data = noise_record.p_signal.transpose()
                        noise_channel = noise_data[channelid%2] # alternate channels of noise to add, per literature

                        newsize = int((len(noise_channel) * downsampled_sampling_rate / record.fs) + 0.5)
                        noise_channel = scipy.signal.resample(noise_channel, newsize)

                        output_dict[subject_str][chname] = {'data' : add_noisy_signal(channel, noise_channel, target_snr_db=int_db), 'labels' : None}
                    elif noise_type=='gn':
                        output_dict[subject_str][chname] = {'data' : add_noise(signal=channel, method='gaussian', target_snr_db=int_db), 'labels' : None}
        
        
        # noise_type=='all':
        noise_type = 'all'
        for dB in ['_6','00','06','12','18','24']:

            int_db = int(dB.replace('_','-'))
            subject_str = subject+noise_type+dB
            output_dict[subject_str] = {}

            # Process each channel separately (2 per input file).
            for channelid, channel in enumerate(data):
                chname = record.sig_name[channelid]
                # Resample from 360Hz to 125Hz
                newsize = int((len(channel) * downsampled_sampling_rate / record.fs) + 0.5)
                channel = scipy.signal.resample(channel, newsize)

                all_noise = np.zeros_like(channel)
                for noise_type_ in ['em', 'ma', 'bw', 'gn']:
                    all_noise += (output_dict[subject+noise_type_+dB][chname]['data'] - channel)
                
                output_dict[subject_str][chname] = {'data' : add_noisy_signal(channel, all_noise, target_snr_db=int_db), 'labels' : None}
    
    return output_dict
    


def load_mitdb(data_path, verbose=False, downsampled_sampling_rate=125):
    """ 
    https://physionet.org/content/mitdb/1.0.0/

    data_path = path where the .dat, .hea, .atr files are located for every patient
    """
    
    realbeats = ['L','R','B','A','a','J','S','V','r',
                'F','e','j','n','E','/','f','Q','?']

    normalbeats = ['N']

    # Loop through each input file. Each file contains one
    # record of ECG readings, sampled at 360 readings per
    # second.

    # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)                  

    subjects = [
        '100','101','102','103','104', '105','106','107','108','109',
        '111','112','113','114','115', '116','117','118','119','121',
        '122','123','124','200','201', '202','203','205','207','208',
        '209','210','212','213','214', '215','217','219','220','221',
        '222','223','228','230','231', '232','233','234']

    # subjects = [
    #     '100',]

    output_dict = {}
    for subject in tqdm(subjects):
        output_dict[subject] = {}

        # Read in the data
        record = wfdb.rdrecord(data_path+subject)
        annotation = wfdb.rdann(data_path+subject, 'atr')

        if verbose:
            # Print some meta informations
            print('Sampling frequency used for this record:', record.fs)
            print('Shape of loaded data array:', record.p_signal.shape)
            print('Number of loaded annotations:', len(annotation.num))
        
        # Get the ECG values from the file.
        data = record.p_signal.transpose()

        # Generate the classifications based on the annotations.
        # -1.0 = undetermined
        # 0.0 = normal
        # 1.0 = abnormal    
        rate = np.zeros_like(annotation.symbol, dtype='float')
        rate[~np.isin(annotation.symbol, normalbeats+realbeats)] = -1.0
        rate[np.isin(annotation.symbol, realbeats)] = 1.0
        rate[np.isin(annotation.symbol, normalbeats)] = 0.0

        rates = np.zeros_like(data[0], dtype='float')
        rates[annotation.sample] = rate

        
        # Process each channel separately (2 per input file).
        for channelid, channel in enumerate(data):
            chname = record.sig_name[channelid]
            # Resample from 360Hz to 125Hz
            newsize = int((len(channel) * downsampled_sampling_rate / record.fs) + 0.5)
            channel = scipy.signal.resample(channel, newsize)
                 
            if verbose:
                print('Name', subject, 'ECG channel type:', chname)
            
            output_dict[subject][chname] = {'data' : channel, 'labels' : rates}
    
    return output_dict


def load_picc(data_path, verbose=True):
    """ 
    https://physionet.org/content/challenge-2011/1.0.0/

    data_path = path where the .dat, .hea, .atr files are located for every patient
    load_method='beat_by_beat' either load and process data beat by beat, or by 'windows'
    
    Note: Different featurizations may be used for each sqi
    """
        
    # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)                  

    normal_subjects = open(data_path+'set-a/RECORDS-acceptable', 'r').readlines()
    normal_subjects = [subj.replace('\n','') for subj in normal_subjects]
    normal_labels = [0.0 for _ in range(len(normal_subjects))]

    abnormal_subjects = open(data_path+'set-a/RECORDS-unacceptable', 'r').readlines()
    abnormal_subjects = [subj.replace('\n','') for subj in abnormal_subjects]
    abnormal_labels = [1.0 for _ in range(len(abnormal_subjects))]

    all_subjects = normal_subjects + abnormal_subjects
    all_labels = normal_labels + abnormal_labels

    output_dict = {}
    for subject, label in tqdm(zip(all_subjects, all_labels)):
        output_dict[subject] = {}

        # Read in the data
        record = wfdb.rdrecord(data_path+'set-a/'+subject)

        # Print some meta informations
        if verbose:
            print('Sampling frequency used for this record:', record.fs)
            print('Shape of loaded data array:', record.p_signal.shape)
        
        # Get the ECG values from the file.
        data = record.p_signal.transpose()
        
        # Process each channel separately (12 per input file).
        for channelid, channel in enumerate(data):
            newsize = int((len(channel) * 125 / record.fs) + 0.5)
            channel = scipy.signal.resample(channel, newsize)

            chname = record.sig_name[channelid]
            if verbose:
                print('ECG channel type:', chname)
            output_dict[subject][chname] = {'data':channel, 'label':label}
    
    return output_dict
