import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# # float_arr = list()
with open("dsss_chann_reverted_output.32cf") as fp:
    dsss_output_array = np.fromfile(fp, dtype='complex64')

# with open("lpi_chann_output.32cf") as fp_:
#     lpi_chann_array = np.fromfile(fp_, dtype='complex64')
# print(len(output_array))

# dsss_barr = dsss_output_array.reshape((1024, 65536))
# dsss_barr = np.fft.fftshift(np.abs(dsss_barr), axes=0)

# lpi_barr  = lpi_chann_array.reshape((1024, 65536))
# lpi_barr  = np.fft.fftshift(np.abs(lpi_barr), axes=0)

fig, ax = plt.subplots()
f, t, Zxx = sig.stft(dsss_output_array, nperseg=1024)

ax.imshow(np.abs(np.fft.fftshift(Zxx, axes=0).T[:40000,]), aspect='auto', origin='lower')

fig.savefig('reverted_stft.png')


# ax.imshow(lpi_barr[:, :20000].T, aspect='auto', origin='lower')
# fig.savefig('Channelized_LPI_combined.png')
# ax.clear()
# Y = np.log10(dsss_barr[:, :40000])
# ax.imshow(Y.T, aspect='auto', origin='lower', vmin=0.5)
# # ax.imshow(np.abs(Zxx.T), aspect='auto')
# fig.savefig('Channelized_DSSS.png')# float_arr = list()
# with open("dsss_chann_output.32cf") as fp:
#     dsss_output_array = np.fromfile(fp, dtype='complex64')

# with open("lpi_chann_output.32cf") as fp_:
#     lpi_chann_array = np.fromfile(fp_, dtype='complex64')
# # print(len(output_array))

# dsss_barr = dsss_output_array.reshape((1024, 65536))
# dsss_barr = np.fft.fftshift(np.abs(dsss_barr), axes=0)

# lpi_barr  = lpi_chann_array.reshape((1024, 65536))
# lpi_barr  = np.fft.fftshift(np.abs(lpi_barr), axes=0)

# fig, ax = plt.subplots()

# ax.imshow(lpi_barr[:, :20000].T, aspect='auto', origin='lower')
# fig.savefig('Channelized_LPI_combined.png')
# ax.clear()
# Y = np.log10(dsss_barr[:, :40000])
# ax.imshow(Y.T, aspect='auto', origin='lower', vmin=0.5)
# # ax.imshow(np.abs(Zxx.T), aspect='auto')
# fig.savefig('Channelized_DSSS.png')