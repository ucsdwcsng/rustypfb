import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# # float_arr = list()
with open("dsss_chann_reverted_output.32cf") as fp:
    dsss_output_array = np.fromfile(fp, dtype='complex64')

with open("lpi_chann_reverted_output.32cf") as fp_:
    lpi_output_array = np.fromfile(fp_, dtype='complex64')
# print(len(output_array))

# dsss_barr = dsss_output_array.reshape((1024, 65536))
# dsss_barr = np.fft.fftshift(np.abs(dsss_barr), axes=0)

# lpi_barr  = lpi_chann_array.reshape((1024, 65536))
# lpi_barr  = np.fft.fftshift(np.abs(lpi_barr), axes=0)

fig, ax = plt.subplots()
f, t, Zxx = sig.stft(lpi_output_array, nperseg=1024, noverlap=512)

ax.imshow(np.abs(np.fft.fftshift(Zxx, axes=0).T[:20000,]), aspect='auto', origin='lower')

fig.savefig('reverted_stft_lpi.png')

ax.clear()

f, t, Zxx = sig.stft(dsss_output_array, nperseg=1024, noverlap=512)

ax.imshow(np.log10(np.abs(np.fft.fftshift(Zxx, axes=0).T[:40000,])), aspect='auto', origin='lower', vmin=0.1)

fig.savefig('reverted_stft_dsss.png')


# N, Np = 8192, 128
# reductor_size = 2*N - Np // 2

# oned_outp_sum = np.fromfile("dsss_original_sum.32f", dtype='float32')
# oned_revert_outp_sum = np.fromfile("dsss_reverted_sum.32f", dtype='float32')

# Q = np.fft.fftshift(np.fft.fftfreq(N))
# K = np.fft.fftshift(np.fft.fftfreq(Np))
# cycles     = np.zeros(reductor_size, dtype='float32')
# for ind in range(reductor_size):
#     reduced_index           =  ind if (ind < N) else (ind - N + (Np // 2))
#     cycles[ind]        =  (Q[reduced_index] + K[0]) if (ind < N) else (Q[reduced_index] + K[-1])
# FIG, (AX1, AX2) = plt.subplots(2, 1)

# AX1.plot(cycles, oned_outp_sum)
# AX1.set_title('original cycle features')
# AX2.plot(cycles, oned_revert_outp_sum)
# AX2.set_title('Reverted cycle features')
# FIG.tight_layout()
# FIG.savefig('Cycle_Features_Under_Reversion.png')


# ax.imshow(lpi_barr[:, :20000].T, aspect='auto', origin='lower')
# fig.savefig('Channelized_LPI_combined.png')
# ax.clear()
# Y = np.log10(dsss_barr[:, :40000])
# ax.imshow(Y.T, aspect='auto', origin='lower', vmin=0.5)
# # ax.imshow(np.abs(Zxx.T), aspect='auto')
# fig.savefig('Channelized_DSSS.png ax_2.plot(cycles, oned_outp_max)
    # ax_2.set_title('Non Conjugate 1D Max reduction')

    # ax_4.plot(cycles, oned_outp_sum)
    # ax_4.set_title('Non Conjugate 1D Sum reduction')
    
    # ')# float_arr = list()
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