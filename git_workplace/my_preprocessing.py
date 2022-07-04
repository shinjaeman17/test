"""
heart sound murmur detetion 전처리 함수 모음
"""

# 필요한 라이브러리 불러오기
import numpy as np
from scipy import signal
import warnings
warnings.filterwarnings('ignore')
"""
normalization, spike removal
"""
def normalization(wave): # 정규화
    normalized_wave = wave / np.max(np.abs(wave))
    return normalized_wave
def schmidt_spike_removal(original_signal, fs = 4000):
    windowsize = int(np.round(fs/4))
    trailingsamples = len(original_signal) % windowsize
    sampleframes = np.reshape(original_signal[0 : len(original_signal)-trailingsamples], (-1, windowsize) )
    MAAs = np.max(np.abs(sampleframes), axis = 1)
    while len(np.where(MAAs > np.median(MAAs)*3 )[0]) != 0:
        window_num = np.argmax(MAAs)
        spike_position = np.argmax(np.abs(sampleframes[window_num,:]))
        zero_crossing = np.abs(np.diff(np.sign(sampleframes[window_num, :])))
        if len(zero_crossing) == 0:
            zero_crossing = [0]
        zero_crossing = np.append(zero_crossing, 0)
        if len(np.nonzero(zero_crossing[:spike_position+1])[0]) > 0:
            spike_start = np.nonzero(zero_crossing[:spike_position+1])[0][-1]
        else:
            spike_start = 0
        zero_crossing[0:spike_position+1] = 0
        spike_end = np.nonzero(zero_crossing)[0][0]
        sampleframes[window_num, spike_start : spike_end] = 0.0001;
        MAAs = np.max(np.abs(sampleframes), axis = 1)
    despiked_signal = sampleframes.flatten()
    despiked_signal = np.concatenate([despiked_signal, original_signal[len(despiked_signal) + 1:]])
    return despiked_signal
"""
Get Wave
"""
def get_wave_features(wave, Fs = 4000, featuresFs = 2000, low = 25, high = 400):
    filtered = wave.copy()
    time = len(wave) / Fs
    n_sample = int(time * featuresFs)
    # Spike removal
    try:
        filtered = schmidt_spike_removal(filtered, fs = 4000)
    except:
        pass
    #filtered = signal.resample(filtered, n_sample)
    filtered = normalization(filtered)
    """ Extract Envelope Features """
    features = np.zeros((4000, 1))
    features[:, 0] = filtered
    return features