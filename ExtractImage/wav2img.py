import librosa
import os
from scipy.fftpack import fft
from scipy.signal import get_window
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import mytools

def wav2img(source_dir,dir,file_name):
    mel_spec, sample_rate = librosa.load(dir+file_name, sr=None)
    # # clip = clip[:132300] # first three seconds of file
    # print(clip)
    # plt.plot([i for i in range(len(clip))], clip)
    # plt.show()
    n_fft = 1024  # frame length


    # start = 45000 # start at a part of the sound thats not silence
    # x = clip[start:start+n_fft]

    # window = get_window('hann', n_fft)
    # x = x * window

    # X = fft(x, n_fft)
    # X = X[:n_fft//2+1]
    # X_magnitude, X_phase = librosa.magphase(X)
    # plt.plot(X_magnitude)
    # X_magnitude_db = librosa.amplitude_to_db(X_magnitude)
    # print(X_magnitude_db)
    # plt.plot( X_magnitude_db)
    # plt.show()

    hop_length = 512
    # stft = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length)
    # stft_magnitude, stft_phase = librosa.magphase(stft)
    # stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    # print(stft_magnitude_db.shape)

    # plt.figure(figsize=(12, 6))
    # librosa.display.specshow(stft_magnitude_db, x_axis='time', y_axis='linear',
    #                          sr=sample_rate, hop_length=hop_length)
    # plt.show()


    n_mels = 64
    f_min = 20
    f_max = 8000
    mel_spec = librosa.feature.melspectrogram(mel_spec, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, sr=sample_rate, power=1.0, fmin=f_min, fmax=f_max)
    mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)

    # info = {
    #     "mel_spec":mel_spec,
    #     "sample_rate":sample_rate
    # }
    # mytools.save_to_npy(info, "../img_data/"+source_dir+"/"+file_name.replace("wav","npy"))
    # # print(mel_spec_db.shape)
    # mytools.save_to_npy(mel_spec_db,"../"+source_dir+"/"+file_name.replace("wav","npy"))
    # exit(0)
    axes,tmp = librosa.display.specshow(mel_spec, x_axis='time',  y_axis='mel',
                             sr=sample_rate, hop_length=hop_length,
                             fmin=f_min, fmax=f_max)
    print(source_dir+"/"+file_name.replace("wav","jpg"))
    plt.savefig(source_dir.replace("filter_data","image_data")+"/"+file_name.replace("wav","jpg"))
    # del(tmp)

if __name__ == "__main__":
    #-----------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------
    # source dir
    ls = ["../data/filter_data/test","../data/filter_data/train/1","../data/filter_data/train/2", "../data/filter_data/train/3", "../data/filter_data/train/4"]
    # ls = ["train/3"]

    for source_dir in ls:
        dir = source_dir+"/"
        #-----------------------------------------------------------------------------------------
        # get *.wav
        wavs = [wav for wav in os.listdir(dir) if "enh_" in wav]

        for i,wav in enumerate(wavs):
            wav2img(source_dir,dir,wav)
            # print(source_dir.replace("filter_data","image_data"),":",i,"/",len(wavs))
