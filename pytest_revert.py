import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

with open("dsss_chann_reverted_output.32cf") as fp1:
    dsss_revert = np.fromfile(fp1, dtype='complex64')

with open("lpi_chann_reverted_output.32cf") as fp2:
    lpi_revert = np.fromfile(fp2, dtype='complex64')

with open("lpi_chann_output.32cf") as fp3:
    lpi_channogram = np.fromfile(fp3, dtype='complex64')

with open("dsss_chann_output.32cf") as fp4:
    dsss_channogram = np.fromfile(fp4, dtype='complex64')

fig, ax = plt.subplots(1, 2)

f, t, Zxx = sig.stft(lpi_revert, nperseg=1024, noverlap=512)
lpi_barr  = lpi_channogram.reshape((1024, 65536))
lpi_barr  = np.abs(lpi_barr)

ax[0].imshow(np.abs(np.fft.fftshift(Zxx, axes=0).T)[:20000,], aspect='auto', origin='lower')
ax[0].set_title('Reverted Spectrum')

ax[1].imshow(lpi_barr.T[:20000,], aspect='auto', origin='lower')
ax[1].set_title('Forward Channogram')
fig.tight_layout()
fig.savefig('LPI.png')

ax[0].clear()
ax[1].clear()

f, t, Zxx = sig.stft(dsss_revert, nperseg=1024, noverlap=512)
dsss_barr  = dsss_channogram.reshape((1024, 65536))
dsss_barr  = np.abs(dsss_barr)

ax[0].imshow(np.log10(np.abs(np.fft.fftshift(Zxx, axes=0).T)[:40000,]), aspect='auto', origin='lower', vmin=0.6)
ax[0].set_title('Reverted Spectrum')

ax[1].imshow(np.log10(dsss_barr.T)[:40000,], aspect='auto', origin='lower', vmin=0.5)
ax[1].set_title('Forward Channogram')
fig.tight_layout()
fig.savefig('DSSS.png')
