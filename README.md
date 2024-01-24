This is an implementation of a Non-maximally decimated Polyphase Filter Bank (PFB) which is callable from the Rust programming Language. This filter bank supports perfect reconstruction. The backend is written in CUDA C++ for real time deployment. 

The image below illustrates the frequency responses of both the analysis and synthesis prototype filters used in the filter bank. The analysis filter is a simple Nyquist filter, and the passband of the synthesis filter is designed to completely cover the transition band of the analysis filter. This ensures distortion free reconstruction.

![Image Alt Text](/docs/filter_responses.png)

For alias free reconstruction, we need minimal overlap between the passbands of the untranslated synthesis filter and the analysis filter when it is shifted by the downsampling factor. 

![Image Alt Text](/docs/shifted_filter_responses.png)

This concludes the theoretical justification for the choices of prototype synthesis and analysis filters. More details on how to arrive at these conditions maybe found in this [paper](https://ieeexplore.ieee.org/document/6690219).

Cosider the following figure for the dataset where many energies are crowded together to create a busy spectrum (we call this the LPI combined example).
![Image Alt Text](/docs/LPI.png)

Depicted here, is the comparison of the reverted spectrum (i.e., the stft of the reverted IQ) with the channogram obtained by forward channelization by the PFB. The prototype analysis and synthesis filters are chosen as above. 

Here is the same comparison for the case where three different DSSS transmissions occur in the spectrum.
![Image Alt Text](/docs/DSSS.png)

In both the cases, we see no aliasing artifacts, and no distortions in the reverted spectrum. By definition, this is perfect reconstruction!

The Rust call signatures are self-documented in the ``channelizer`` crate. The ``examples`` directory here shows how both the forward and revert work.

The forward channelizer process function has been benchmarked to attain a throughput of ``~763 Megasamples per second`` on a single NVIDIA A10 GPU core. 

In order to check things are running as required, the test written in the root of the ``channelizer`` crate creates channelized output. One can visualize the output channogram by running the ``pytest_lib_visual.py`` script. 

To test that the reverse channelization is working as expected, run the ``revert_example.rs`` inside examples. To visualize the result, run ``pytest_revert.py`` to generate the png files linked here.

All dependencies are listed in the Dockerfile included in the repo.

Happy channelizing!

