import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# float_arr = list()
with open("chann_output.32cf") as fp:
    output_array = np.fromfile(fp, dtype='complex64')

# print(len(output_array))

barr = output_array.reshape((1024, 262144))
barr = np.fft.fftshift(np.abs(barr), axes=0)
# print(barr)
# # # barr = np.abs(arr)
# # # f, t, Zxx = sig.stft(arr, window=sig.windows.kaiser(beta=10.0, M=1024), nperseg=1024, noverlap=512)
fig, ax = plt.subplots()
# print(np.max(barr), np.min(barr), np.median(barr))
# # # ax.plot(np.log10(barr).T)
# # # # ax.plot(np.abs(np.fft.fftshift(Zxx, axes=1)))
# pos = ax.imshow(np.log10(barr)[:, 0:40000], vmin=-3, vmax=1, aspect='auto', origin='lower')
# # # # # plt.colorbar()
# fig.colorbar(pos, ax=ax)
# fig.savefig('busyBandDSSS_channelized_wrong_reshape.png')
# exit(-1)
# # f = np.fft.fftshift(np.arange(1024))
# # print(len(t))
# # print(len(arr))
# # print(barr)
# # freqs = np.fft.fftshift(np.fft.fftfreq(1024))
# # nslice = np.array(np.arange(32764*2))
# Zxx = np.abs(np.fft.fftshift(barr, axes=1))
# print(np.max(Zxx), np.min(Zxx))
# fig, ax = plt.subplots()
Y = np.log10(barr[:, :40000])
ax.imshow(Y, aspect='auto', vmin=-2.4)
# ax.plot(Zxx.T)
fig.savefig('busyBandDSSS_Channelized.png')