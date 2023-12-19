This is an implementation of a Non-Maximally decimated Polyphase Filter Bank (PFB) which is callable from the Rust programming Language. This filter bank supports perfect reconstruction. The backend is written in CUDA C++ for real time deployment. 

The image below illustrates the frequency responses of the analysis and synthesis filters. The analysis filter is a simple Nyquist filter, and the passband of the synthesis filter is designed to completely cover the transition band of the analysis filter, as required for perfect reconstruction.

Usage instructions coming soon.

![Image Alt Text](/docs/filter_responses.png)

