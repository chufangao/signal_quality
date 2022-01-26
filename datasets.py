import numpy as np
import wfdb
import matplotlib.pyplot as plt
import biosppy 
import scipy
from tqdm import tqdm


def process_ecg_beat_by_beat(channel, rates, sampling_rate, downsampled_sampling_rate=125, beat_window=90, show=False):
    ### channel is the entire raw ECG waveform of shape (length,)
    ### rates is the processed (numeric) annotations of same shape as ECG (length,)
    ### sampling rate is the hz of the ECG signal
    
    # Resample from 360Hz to 125Hz
    newsize = int((len(channel) * downsampled_sampling_rate / sampling_rate) + 0.5)
    channel = scipy.signal.resample(channel, newsize)

    # Find rpeaks in the ECG data. Most should match with
    # the annotations.
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
        catval = rates[max(0,idxval-10):idxval+10].max()
        
        # Skip beat if there is no classification.
        if catval == -1.0:
            continue

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

        # # Append the classification to the beat data.
        # beats[idx] = np.append(beats[idx], catval)
        labels.append(catval)
        beats.append(beat)

    # return data and labels
    return beats, labels


def load_mit_bih(data_path='/zfsauton/project/public/chufang/mit-bih-arrhythmia-database-1.0.0/'):
    # Instead of using the annotations to find the beats, we will
    # use R-peak detection instead. The reason for this is so that
    # the same logic can be used to analyze new and un-annotated
    # ECG data. We use the annotations here only to classify the
    # beat as either Normal or Abnormal and to train the model.
    # Reference:
    # https://physionet.org/physiobank/database/html/mitdbdir/intro.htm
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
            print('ECG channel type:', chname)
            
            data, labels = process_ecg_beat_by_beat(channel=channel, rates=rates, sampling_rate=record.fs,
                                                    downsampled_sampling_rate=125, beat_window=90, show=False)
            # Save to CSV file.
            output_dict[subject][chname] = {'data':data, 'labels':labels}
            # savedata = np.array(list(beats[:]), dtype=np.float)
    
    return output_dict
