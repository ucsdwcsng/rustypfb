This is an implementation of a Non-Maximally decimated Polyphase Filter Bank (PFB) which is callable from the Rust programming Language. This filter bank supports perfect reconstruction. The backend is written in CUDA C++ for real time deployment. 

The image below illustrates the frequency responses of both the analysis and synthesis prototype filters used in the filter bank. The analysis filter is a simple Nyquist filter, and the passband of the synthesis filter is designed to completely cover the transition band of the analysis filter. This ensures alias cancellation.

![Image Alt Text](/docs/filter_responses.png)

For distortion free reconstruction, we need minimal overlap between the passbands of the analysis filter when it is shifted by the downsampling factor, and the synthesis filter. 

![Image Alt Text](/docs/shifted_filter_responses.png)

This concludes the theoretical justification for the choices of prototype synthesis and analysis filters.

Let's take a look at some examples of forward channelization with the PFB. Depicted here is the channogram for the LPI combined scenario when Nch = 1024.

![Image Alt Text](/docs/Channelized_LPI_combined.png)

Next, here is a constructed channogram for the DSSS scenario 

![Image Alt Text](/docs/Channelized_LPI_combined.png)

Usage instructions coming soon.



