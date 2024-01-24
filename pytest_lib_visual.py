import numpy as np
import matplotlib.pyplot as plt

with open("dsss_chann_output.32cf") as fp:
    dsss_output_array = np.fromfile(fp, dtype='complex64')

with open("lpi_chann_output.32cf") as fp_:
    lpi_chann_array = np.fromfile(fp_, dtype='complex64')

dsss_barr = dsss_output_array.reshape((1024, 65536))
dsss_barr = np.abs(dsss_barr)

lpi_barr  = lpi_chann_array.reshape((1024, 65536))
lpi_barr  = np.abs(lpi_barr)

fig, ax = plt.subplots()

ax.imshow(lpi_barr[:, :20000].T, aspect='auto', origin='lower')
fig.savefig('Channelized_LPI_combined.png')
ax.clear()
Y = np.log10(dsss_barr[:, :40000])
ax.imshow(Y.T, aspect='auto', origin='lower', vmin=0.5)
fig.savefig('Channelized_DSSS.png')
