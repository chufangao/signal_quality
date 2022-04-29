### imports
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from datetime import datetime
import neurokit2 as nk
from tqdm import trange, tqdm
import multiprocessing
import pickle
sys.path.append('../')
from signal_quality import featurization, sqis

def helper(window):
    if len(window) < 125:
        return [np.nan] * 130

    pleth_raw = window['Pleth'].values
    pleth_cleaned = nk.ppg_clean(ppg_signal=pleth_raw, sampling_rate=sampling_rate, method='elgendi')
    pleth_features = featurization.featurize_pleth(window=pleth_cleaned, pleth_time=window.index.values) + \
        sqis.get_pleth_sqis(pleth_raw, pleth_cleaned)

    ecg_raw = window['ECG'].values
    ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=sampling_rate, method="neurokit")
    peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='kalidas2017')[1]['ECG_R_Peaks']
    ecg_features = featurization.featurize_ecg(window=ecg_cleaned, sampling_rate=sampling_rate) + \
        sqis.get_ecg_sqis(ecg_raw, ecg_cleaned, peaks, sampling_rate, window=step_size)

    cco_features = sqis.get_generic_sqis(window['CCO'].values)
    svo2_features = sqis.get_generic_sqis(window['SvO2'].values)
    spo2_features = sqis.get_generic_sqis(window['SpO2'].values)
    pap_features = sqis.get_generic_sqis(window['PAP'].values)
    cvp_features = sqis.get_generic_sqis(window['CVP'].values)
    art_features = sqis.get_generic_sqis(window['ART'].values)
    airpr_features = sqis.get_generic_sqis(window['AirPr'].values)

    return pleth_features + ecg_features + cco_features + svo2_features + spo2_features + \
        pap_features + cvp_features + art_features + airpr_features


if __name__=='__main__':

    h5_paths = "/zfsauton/data/public/vleonard/tracir/auv_files/"
    # paths = os.listdir(h5_paths)
    # paths.remove('p06_microtrend_data.xlsx')
    # paths = sorted(paths)
    paths = ['p01.h5', 'p02.h5', 'p03.h5', 'p04.h5', 'p05.h5', 
        'p06.h5', 'p07.h5', 'p08.h5', 'p09.h5', 'p10.h5', 
        'p11.h5', 'p12.h5', 'p13.h5', 'p14.h5', 'p15.h5',
        'p16.h5', 'p17.h5', 'p18.h5', 'p19.h5', 'p20.h5', 
        'p21.h5', 'p22.h5', 'p23.h5', 'p24.h5', 'p25.h5', 
        'p26.h5']
    print(paths)

    path_dict = {path: {} for path in paths}
    # [1,2,3,4,7,8]
    # [2,3,4,7,8] - [1,2,3,4,7]

    sampling_rate = 125
    step_size = 10 # seconds
    for path in paths[:]:
        f = h5py.File(h5_paths+path, 'r')
        data = pd.DataFrame(f['daq'][()])

        for bad_col in ['ECG_SQ', 'XXXX', 'Clicktrack']:
            if bad_col in data.columns:
                data.drop(columns=[bad_col], inplace=True)

        print(path, data.columns, data.shape)

        data['timestamp'] -= np.min(data['timestamp'])
        data.set_index('timestamp', inplace=True)

        start = data.index[0]
        end = data.index[-1]

        all_features = []

        with multiprocessing.Pool(processes=16) as pool:
        
            args = [data.iloc[(i<data.index) & (data.index<i+step_size)] for i in range(int(start), int(end), step_size)]
            all_features_ = list(tqdm(pool.imap(helper, args), total=len(args)))
            all_features.append(all_features_)

        np.save(file=path[:-3]+'.npy', arr=np.array(all_features))
