# Signal Quality
Collection of tools to analyze ECG, Plethysmography, and general time-series signal quality

[**Documentation**](https://chufangao.github.io/signal_quality/) 
[Github](https://github.com/chufangao/signal_quality/)

### Main Submodules
- **datasets** - Code to load example ECG signal quality datasets, including the The PhysioNet/Computing in Cardiology Challenge 2011 Dataset and the MIT-BIH Arrhythmia Database.
- **sqis** - Functions that calculate signal quality indicies on time-series signals. Most implementations are focused on ECG quality, as it is a common area of research.
- **featurization** - Contains functions that calculate geometric waveform features of Pleth and ECG.
