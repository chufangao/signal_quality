import torch
from tqdm import tqdm
import pywt
import matplotlib.pyplot as plt
import emd
import numpy as np
import scipy
import sys
import emd
import neurokit2 as nk
import datasets

# print(torch.__version__)

### ===================================================== Neural Networks =====================================================

class Net(torch.nn.Module):
    def __init__(self, h_sizes=[768, 256, 64, 2], activation=torch.nn.LeakyReLU()):
        super(Net, self).__init__()
        """
        Flexible feed forward network over a base encoder. 
        
        Parameters
        ---------- 
        encoder_model: The base encoder model
        h_sizes: Linear layer sizes to be used in the MLP
        activation: Activation function to be use in the MLP. 
        device: Device to use for training. 'cpu' by default.
        """

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.layers = torch.nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.layers.append(torch.nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.layers.append(activation)
            # self.layers.append(torch.nn.Dropout(p=0.5))
        self.to(self.device)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)    
        return x

    def fit(self, x_train, y_train, batch_size=32, epochs=10, criterion=torch.nn.MSELoss(), lr=1e-3, weight_decay=1e-4, shuffle=True, seed=0):
        assert type(x_train)==np.ndarray and type(y_train)==np.ndarray

        np.random.seed(seed)

        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        print(self.device)
        for epoch in tqdm(range(epochs)):
            if shuffle:
                inds = np.arange(len(x_train))
                np.random.shuffle(inds)
                x_train = x_train[inds]
                y_train = y_train[inds]

            for i in range(0, len(x_train), batch_size):
                batch_x = x_train[i:i+batch_size].to(self.device)
                batch_y = y_train[i:i+batch_size].to(self.device)
                            
                out = self.forward(batch_x)
                loss = criterion(out, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    tqdm.write("loss: {:.6f}".format(loss.detach().cpu().numpy()))

    def predict(self, x, batch_size=200):
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            out = []
            
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size].to(self.device)
                out.append(self.forward(batch_x).cpu().numpy())
            return np.concatenate(out)

class LSTMNet(Net):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1, activation=torch.nn.LeakyReLU()):
        super(LSTMNet, self).__init__()
        """
        Flexible feed forward network over a base encoder. 
        
        Parameters
        ---------- 
        encoder_model: The base encoder model
        h_sizes: Linear layer sizes to be used in the MLP
        activation: Activation function to be use in the MLP. 
        device: Device to use for training. 'cpu' by default.
        """
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.activation = activation
        
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

        self.to(self.device)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # out, (hn, cn) = self.lstm(x, (h0, c0))
        # out shape is (batch_size, seq_len, hidden_dim)
        
        # Index hidden state of last time step
        # print(out.shape)
        out = self.activation(out)
        out = self.fc(out)
        # out shape is (batch_size, seq_len, 1)
        out = out.squeeze()
        # out shape is (batch_size, seq_len)
        return out
 
class Conv1DNet(Net):
    def __init__(self, h_sizes=[6, 768, 256, 64, 1], activation=torch.nn.LeakyReLU()):
        super(Conv1DNet, self).__init__()

        self.layers = torch.nn.ModuleList()
        for k in range(len(h_sizes)-1):
            # self.layers.append(torch.nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.layers.append(torch.nn.Conv1d(in_channels=h_sizes[k], out_channels=h_sizes[k+1], kernel_size=5, padding='same'))
            self.layers.append(activation)
            if k != len(h_sizes)-1:
                self.layers.append(torch.nn.BatchNorm1d(h_sizes[k+1]))

            # self.layers.append(torch.nn.Dropout(p=0.5))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)


    def forward(self, x):
        x = torch.permute(x, (0,2,1))
        for layer in self.layers:
            x = layer(x)    
        return x.squeeze()

class Conv1DAutoencoderNet(Net):
    def __init__(self, h_size, input_len, activation=torch.nn.ELU(), kernel_size=8, stride=2):
        super().__init__()
        
        pad = (((stride-1)*input_len-stride+kernel_size) // 2)

        self.encoder = torch.nn.ModuleList()
        self.encoder.append(torch.nn.Conv1d(in_channels=h_size, out_channels=40, kernel_size=kernel_size, stride=stride, padding=pad))
        self.encoder.append(activation)
        self.encoder.append(torch.nn.BatchNorm1d(40))
        self.encoder.append(torch.nn.Conv1d(in_channels=40, out_channels=20, kernel_size=kernel_size, stride=stride, padding=pad))
        self.encoder.append(activation)
        self.encoder.append(torch.nn.BatchNorm1d(20))
        self.encoder.append(torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=stride, padding=pad))
        self.encoder.append(activation)
        self.encoder.append(torch.nn.BatchNorm1d(20))
        self.encoder.append(torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=stride, padding=pad))
        self.encoder.append(activation)
        self.encoder.append(torch.nn.BatchNorm1d(20))
        self.encoder.append(torch.nn.Conv1d(in_channels=20, out_channels=40, kernel_size=kernel_size, stride=stride, padding=pad))
        self.encoder.append(activation)
        self.encoder.append(torch.nn.BatchNorm1d(40))
        # self.encoder.append(torch.nn.Conv1d(in_channels=40, out_channels=1, kernel_size=kernel_size, stride=stride, padding=pad))
        # self.encoder.append(activation)
        # self.encoder.append(torch.nn.BatchNorm1d(1))

        self.decoder = torch.nn.ModuleList()
        # self.decoder.append(torch.nn.ConvTranspose1d(in_channels=1, out_channels=2, kernel_size=kernel_size, stride=stride, padding=pad))
        # self.encoder.append(activation)
        # self.decoder.append(torch.nn.BatchNorm1d(1))
        # self.decoder.append(torch.nn.ConvTranspose1d(in_channels=1, out_channels=40, kernel_size=kernel_size, stride=stride, padding=pad))
        # self.encoder.append(activation)
        # self.decoder.append(torch.nn.BatchNorm1d(40))
        self.decoder.append(torch.nn.ConvTranspose1d(in_channels=40, out_channels=20, kernel_size=kernel_size, stride=stride, padding=pad))
        self.encoder.append(activation)
        self.decoder.append(torch.nn.BatchNorm1d(20))
        self.decoder.append(torch.nn.ConvTranspose1d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=stride, padding=pad))
        self.encoder.append(activation)
        self.decoder.append(torch.nn.BatchNorm1d(20))
        self.decoder.append(torch.nn.ConvTranspose1d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=stride, padding=pad))
        self.encoder.append(activation)
        self.decoder.append(torch.nn.BatchNorm1d(20))
        self.decoder.append(torch.nn.ConvTranspose1d(in_channels=20, out_channels=40, kernel_size=kernel_size, stride=stride, padding=pad))
        self.decoder.append(torch.nn.BatchNorm1d(40))
        self.encoder.append(activation)
        self.decoder.append(torch.nn.ConvTranspose1d(in_channels=40, out_channels=1, kernel_size=kernel_size, stride=stride, padding=pad))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
    
    def forward(self, x):

        x = torch.permute(x, (0,2,1))
        for layer in self.encoder:
            x = layer(x)    
        for layer in self.decoder:
            x = layer(x)

        return x.squeeze()

### ===================================================== Utility Functions =====================================================

def decompose_signal(signal, wavelet='db2'):
    wavelet_levels = pywt.dwt_max_level(len(signal), wavelet)
    wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=wavelet_levels)

    # Detail coefficients at each decomposition level
    dcoeffs = coeffs[1:]

    # Estimate the noise via the method in [2]_
    detail_coeffs = dcoeffs[-1]

    # 75th quantile of the underlying, symmetric noise distribution
    denom = scipy.stats.norm.ppf(0.75)
    sigma = np.median(np.abs(detail_coeffs)) / denom

    # # The VisuShrink thresholds from donoho et al.
    # threshold =  sigma*np.sqrt(2*np.log(len(signal)))
    # denoised_detail = [pywt.threshold(data=level, value=threshold, mode='soft') for level in dcoeffs]
    # # print([c.shape[0] for c in denoised_detail], np.sum([c.shape[0] for c in denoised_detail]))
    denoised_detail = [scipy.signal.resample(x=c, num=len(signal)) for c in coeffs]
    # # print(np.stack(denoised_detail, axis=-1).shape)
    denoised_detail = np.stack(denoised_detail, axis=-1)

    return np.concatenate([denoised_detail, signal[:,np.newaxis]], axis=1)

def rescale_signal(x):
    r = np.max(x) - np.min(x)
    if r == 0:
        return np.zeros_like(x)
    else:
        return 2*(x-np.min(x)) / r - 1

def load_data(mitdb_dict, nstdb_extra_dict, window=512, train_end_ind=100000):
    train_x = []
    train_y = []
    train_noise_type = []
    test_x = []
    test_y = []
    test_noise_type = []
    
    # print(len(mitdb_dict[subject]['MLII']['data']))

    for subject in  ['118', '119']:
        for noise_type in ['em', 'ma', 'bw', 'gn', 'all']:
        # for noise_type in ['gn']:
            # for dB in ['_6','00','06','12','18','24']:
            for dB in ['_6','00','06','12','18','24']:
                orig = mitdb_dict[subject]['MLII']['data']
                orig = rescale_signal(orig)
                noisy = nstdb_extra_dict[subject+noise_type+dB]['MLII']['data']
                noisy = rescale_signal(noisy)

                for i in range(0, train_end_ind, window):
                    train_x.append(noisy[i:i+window])
                    train_y.append(orig[i:i+window])
                    train_noise_type.append(noise_type+dB)

                    test_x.append(noisy[train_end_ind+i:train_end_ind+i+window])
                    test_y.append(orig[train_end_ind+i:train_end_ind+i+window])
                    test_noise_type.append(noise_type+dB)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_noise_type = np.array(train_noise_type)

    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_noise_type = np.array(test_noise_type)

    return train_x, train_y, train_noise_type, test_x, test_y, test_noise_type


def denoise_wavelet(signal, wavelet='db4', wavelet_levels=None, mode='soft', thesh_method='donoho', visualize=False):
    """Based off of the official skimage implementation: 
    https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_wavelet
    
    > import skimage
    > temp1 = skimage.restoration.denoise_wavelet(image=example_signal, sigma=None, wavelet='db4', mode='soft', 
    >                                             wavelet_levels=None, convert2ycbcr=False, method='VisuShrink', 
    >                                             rescale_sigma=True, channel_axis=None)
    > temp2 = denoise_wavelet(example_signal)
    > print(np.all(temp1==temp2)) # returns True
    """
    if wavelet_levels is None:
        wavelet_levels = pywt.dwt_max_level(len(signal), wavelet)
        wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=wavelet_levels)

    # Detail coefficients at each decomposition level
    dcoeffs = coeffs[1:]

    # Estimate the noise via the method in [2]_
    detail_coeffs = dcoeffs[-1]

    if thesh_method == 'donoho':
        # 75th quantile of the underlying, symmetric noise distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
        
        # The VisuShrink thresholds from donoho et al.
        threshold =  sigma*np.sqrt(2*np.log(len(signal)))
    else:
        raise NotImplementedError

    if visualize:
        plt.figure(figsize=(15,10))
        for i, level in enumerate(dcoeffs):
            plt.subplot(wavelet_levels, 1, i+1)
            plt.plot(level)
            # plt.plot(pywt.threshold(data=level, value=threshold, mode=mode))
        plt.show()

    denoised_detail = [pywt.threshold(data=level, value=threshold, mode=mode) for level in dcoeffs]

    denoised_coeffs = [coeffs[0]] + denoised_detail
    return pywt.waverec(denoised_coeffs, wavelet)


def denoise_emd(signal, sample_rate, visualize=False, P=5, beta=30):
    """Follows the approach as defined by Weng et al.
    """
    # Predefine P as opposed to t-test

    ecg_cleaned = nk.ecg_clean(signal, sampling_rate=sample_rate, method="neurokit")
    signals, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sample_rate, method='kalidas2017')
    rpeak_mask = signals.values.flatten()
    rpeaks = info['ECG_R_Peaks']

    # print(rpeaks)
    # plt.show()
    # print(len(rpeaks), len(waves_dwt['ECG_R_Offsets']), len(waves_dwt['ECG_R_Onsets']))
    try:
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=sample_rate, method="dwt", show=visualize, show_type='bounds_R')
        flat_window = int(np.nanmean(np.array(waves_dwt['ECG_R_Offsets'])-np.array(waves_dwt['ECG_R_Onsets']))/2)+1
    except Exception as e:
        return signal

    imf = emd.sift.sift(signal)
    if visualize:
        emd.plotting.plot_imfs(imf)

    reconstruction = np.zeros_like(signal)
    P = min(P, imf.shape[1]-1)
    for i in range(P):
        tukey_window = scipy.signal.windows.tukey(M=int(flat_window + flat_window * 2*beta * (i+1)*.01))
        mask = np.convolve(rpeak_mask, tukey_window, mode='same')
        reconstruction += mask * imf[:,i]
    
    if P <= imf.shape[1]:
        reconstruction += imf[:,P:].sum(axis=1)

    return reconstruction

def denoise_nn(signal, model):
    # assume that input signals are of shape (n,512)
    if len(signal.shape) == 1 and len(signal) == 512:
        signals = [decompose_signal(rescale_signal(signal))]
        return model.predict(np.array(signals))
    elif len(signal.shape) == 2 and signal.shape[1] == 512:
        signals = [decompose_signal(rescale_signal(signal_)) for signal_ in signal]
        return model.predict(np.array(signals))
    else:
        raise NotImplementedError
 
def get_power(signal):
    return np.mean(np.power(signal, 2))

def calc_snr(signal, noise):
    signal_avg_watts = get_power(signal)
    noise_avg_watts = get_power(noise)

    return 10*(np.log10(signal_avg_watts) - np.log10(noise_avg_watts))

def calc_snr_ecg(signal, noise, sampling_rate):
    peaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate, method='kalidas2017')[1]['ECG_R_Peaks']
    window = int(sampling_rate*.05)
    peak_ranges = np.array([np.ptp(signal[max(0, peak-window):min(len(signal), peak+window)]) for peak in peaks])
    peak_ranges = peak_ranges[(np.percentile(peak_ranges, 5)<peak_ranges) & (peak_ranges<np.percentile(peak_ranges, 95))]
    S = np.mean(peaks)**2 / 8

    noise_windows = [noise[i:i+sampling_rate] for i in range(0, len(noise), sampling_rate)]
    noise_rmsds = np.array([np.mean(np.power(w-w.mean(),2)) for w in noise_windows])
    noise_rmsds = noise_rmsds[(np.percentile(noise_rmsds, 5)<noise_rmsds) & (noise_rmsds<np.percentile(noise_rmsds, 95))]
    N = np.mean(noise_rmsds)**2
    
    return 10*(np.log10(S) - np.log10(N))


### ===================================================== Driver for training NNs =====================================================
# import argparse

if __name__=="__main__":
    mitdb_dict = datasets.load_mitdb(data_path='/zfsauton/project/public/chufang/mitdb/', verbose=False)
    nstdb_extra_dict = datasets.load_nstdb_extra(nstdb_path='/zfsauton/project/public/chufang/nstdb/', mitdb_path='/zfsauton/project/public/chufang/mitdb/', verbose=False, downsampled_sampling_rate=125)
    train_x, train_y, train_noise_type, test_x, test_y, test_noise_type = load_data(mitdb_dict, nstdb_extra_dict, window=512)
    train_x = np.array([decompose_signal(x_) for x_ in train_x])
    test_x = np.array([decompose_signal(x_) for x_ in test_x])
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    train = False

    ## Training LSTM ---------------------------------------------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    lstm = LSTMNet(input_dim=6, hidden_dim=64, layer_dim=1)

    if train:
        lstm.fit(train_x, train_y, epochs=30, batch_size=16, criterion=torch.nn.MSELoss(reduction='mean'))
        torch.save(lstm.cpu().state_dict(), 'denoising_models/lstm.pt')
    else:
        lstm.load_state_dict(torch.load('denoising_models/lstm.pt'))
    
    predictions = lstm.predict(test_x) 
    np.save('denoising_models/lstm_preds.npy', predictions)

    ## Training 1dCNN ---------------------------------------------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    cnn = Conv1DNet(h_sizes=[6, 256, 128, 64, 64, 1], activation=torch.nn.LeakyReLU())

    if train:
        cnn.fit(train_x, train_y, epochs=30, batch_size=16, criterion=torch.nn.MSELoss(reduction='mean'))
        torch.save(cnn.cpu().state_dict(), 'denoising_models/cnn.pt')
    else:
        cnn.load_state_dict(torch.load('denoising_models/cnn.pt'))

    predictions = cnn.predict(test_x) 
    np.save('denoising_models/cnn_preds.npy', predictions)

    ## Training 1dCNNAutoencoder ---------------------------------------------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    cnn = Conv1DAutoencoderNet(h_size=6, input_len=train_x.shape[1], activation=torch.nn.LeakyReLU(), kernel_size=8, stride=2)

    if train:
        cnn.fit(train_x, train_y, epochs=30, batch_size=16, criterion=torch.nn.MSELoss(reduction='mean'))
        torch.save(cnn.cpu().state_dict(), 'denoising_models/cnnae.pt')
    else:
        cnn.load_state_dict(torch.load('denoising_models/cnnae.pt'))

    predictions = cnn.predict(test_x)
    np.save('denoising_models/cnnae_preds.npy', predictions)
    
    