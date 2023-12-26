This is an implementation of a Non-Maximally decimated Polyphase Filter Bank (PFB) which is callable from the Rust programming Language. This filter bank supports perfect reconstruction. The backend is written in CUDA C++ for real time deployment. 

The image below illustrates the frequency responses of both the analysis and synthesis prototype filters used in the filter bank. The analysis filter is a simple Nyquist filter, and the passband of the synthesis filter is designed to completely cover the transition band of the analysis filter. This ensures distortion free reconstruction.

![Image Alt Text](/docs/filter_responses.png)

For alias free reconstruction, we need minimal overlap between the passbands of the untranslated synthesis filter and the analysis filter when it is shifted by the downsampling factor. 

![Image Alt Text](/docs/shifted_filter_responses.png)

This concludes the theoretical justification for the choices of prototype synthesis and analysis filters. More details on how to arrive at these conditions maybe found in this [paper](https://ieeexplore.ieee.org/document/6690219).

Let's take a look at some examples of forward channelization with the PFB. Depicted here is the channogram for the LPI combined scenario when Nch = 1024.

![Image Alt Text](/docs/Channelized_LPI_combined.png)

Next, here is the constructed channogram for the DSSS scenario 

![Image Alt Text](/docs/Channelized_DSSS.png)

We can plot the simple spectrogram for the same IQ files and obtain the same result as above. Next, to test the reconstruction algorithm, we first channelize the input, perform full revert of the channelized output and then visualize the spectrum of the reverted IQ.

Here is the reverted spectrum for the LPI combined scenario
![Image Alt Text](/docs/reverted_stft_lpi.png)

and the same for the DSSS scenario
![Image Alt Text](/docs/reverted_stft_dsss.png)

We see no aliasing artifacts, and no distortions in the reverted spectrum. Perfect reconstruction!

Next, we discuss the revert algorithm. The synthesizer needs to write the reconstructed IQ samples to an output buffer. The sizes of the boxes to be reverted cannot be known in advance. Therefore, in order to bound the size od the output memory in advance, we need to give a reasonable upper bound on the number of boxes to be reverted. The interesting thing about the following analysis is that it would return a bound on the number of boxes that are 'small' in some sense.

Usage instructions coming soon.



