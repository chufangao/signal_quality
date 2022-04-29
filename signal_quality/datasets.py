import numpy as np
import wfdb
import matplotlib.pyplot as plt
import biosppy 
import scipy
from tqdm import tqdm

def load_ecg_by_windows(channel, rates, sampling_rate, noise_threshold, 
    downsampled_sampling_rate=125, window_size=60*125, step=30*125):
    """
    channel = the entire raw ECG waveform of shape (length,)
    rates = the processed (numeric) annotations of same shape as ECG (length,)
    sampling_rate = the hz of the input ECG signal
    downsampled_sampling_rate = the hz to downsample to
    window_size = the window, after downsampling, of indices in a feature window
    """
    
    # Resample from 360Hz to 125Hz
    newsize = int((len(channel) * downsampled_sampling_rate / sampling_rate) + 0.5)
    channel = scipy.signal.resample(channel, newsize)
    
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

def load_ecg_beat_by_beat(channel, rates, sampling_rate, 
    downsampled_sampling_rate=125, beat_window=90, show=False):
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

    # Resample from 360Hz to 125Hz
    newsize = int((len(channel) * downsampled_sampling_rate / sampling_rate) + 0.5)
    channel = scipy.signal.resample(channel, newsize)

    # Find rpeaks in the ECG data. Most should match with
    # the annotations.
    # biosppy.signals.ecg.ecg returns:
    # names = ('ts', 'filtered', 'rpeaks', 'templates_ts', 'templates',
    #          'heart_rate_ts', 'heart_rate')
    out = biosppy.signals.ecg.ecg(signal=channel, sampling_rate=downsampled_sampling_rate, show=show)
    
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
            plt.legend(loc='upper right'); plt.grid(); plt.ylabel('ECG'); plt.xlabel(str(downsampled_sampling_rate)+' hz'); plt.show()

        # # Normalize the readings to a 0-1 range for ML purposes.
        # beat_range = beat.max() - beat.min()
        # if beat_range == 0:
        #     continue
        # beat = (beat - beat.min()) / beat_range

        beats.append(beat)

    # return data and labels
    return beats, labels


def load_mit_bih(data_path='/zfsauton/project/public/chufang/MIT-BIH/', load_method='beat_by_beat', verbose=False):
    """ 
    https://physionet.org/content/mitdb/1.0.0/

    data_path = path where the .dat, .hea, .atr files are located for every patient
    load_method='beat_by_beat' either load and process data beat by beat, or by 'windows'
    
    Note: Different featurizations may be used for each sqi
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
             
            if verbose:
                print('ECG channel type:', chname)
            
            if load_method=='beat_by_beat':
                data, labels = load_ecg_beat_by_beat(channel=channel, rates=rates, sampling_rate=record.fs,
                                                     downsampled_sampling_rate=125, beat_window=90, show=False)
            elif load_method=='windows':
                data, labels = load_ecg_by_windows(channel=channel, rates=rates, sampling_rate=record.fs, noise_threshold=0.001,
                                                   downsampled_sampling_rate=125, window_size=5*60*125, step=30*125)
                # data, feature_names = ecg_featurization.featurize_windows(windows=data, sampling_rate=125)
            else:
                raise NotImplementedError

            output_dict[subject][chname] = {'data':data, 'labels':labels}
            # savedata = np.array(list(beats[:]), dtype=np.float)
    
    return output_dict

def load_picc(data_path='/zfsauton/project/public/chufang/PICC/', verbose=True):
    """ 
    https://physionet.org/content/challenge-2011/1.0.0/

    data_path = path where the .dat, .hea, .atr files are located for every patient
    load_method='beat_by_beat' either load and process data beat by beat, or by 'windows'
    
    Note: Different featurizations may be used for each sqi
    """
        
    # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)                  

    data_path='/zfsauton/project/public/chufang/PICC/'
    normal_subjects = open(data_path+'set-a/RECORDS-acceptable', 'r').readlines()
    normal_subjects = [subj.replace('\n','') for subj in normal_subjects]
    labels = [0.0 for _ in range(len(normal_subjects))]

    abnormal_subjects = open(data_path+'set-a/RECORDS-unacceptable', 'r').readlines()
    abnormal_subjects = [subj.replace('\n','') for subj in abnormal_subjects]
    labels = labels + [1.0 for _ in range(len(abnormal_subjects))]

    output_dict = {}
    for subject, label in tqdm(zip(normal_subjects + abnormal_subjects, labels)):
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
            output_dict[subject][chname] = {'data':channel, 'labels':[label]}
            # savedata = np.array(list(beats[:]), dtype=np.float)
    
    return output_dict
