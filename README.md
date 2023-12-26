This is an implementation of a Non-Maximally decimated Polyphase Filter Bank (PFB) which is callable from the Rust programming Language. This filter bank supports perfect reconstruction. The backend is written in CUDA C++ for real time deployment. 

The image below illustrates the frequency responses of both the analysis and synthesis prototype filters used in the filter bank. The analysis filter is a simple Nyquist filter, and the passband of the synthesis filter is designed to completely cover the transition band of the analysis filter. This ensures distortion free reconstruction.

![Image Alt Text](/docs/filter_responses.png)

For alias free reconstruction, we need minimal overlap between the passbands of the untranslated synthesis filter and the analysis filter when it is shifted by the downsampling factor. 

![Image Alt Text](/docs/shifted_filter_responses.png)

This concludes the theoretical justification for the choices of prototype synthesis and analysis filters. More details on how to arrive at these conditions maybe found in this [paper](https://ieeexplore.ieee.org/document/6690219).

Cosider the following figure for the LPI combined dataset.
![Image Alt Text](/docs/LPI.png)

We are comparing the reverted spectrum (i.e., the stft of the reverted IQ) with the channogram obtained by forward channelization by the PFB. The prototype analysis and synthesis filters are chosen as above. 

Here is the same comparison with the DSSS scenario.
![Image Alt Text](/docs/DSSS.png)

In both the cases, we see no aliasing artifacts, and no distortions in the reverted spectrum. By definition, this is perfect reconstruction!

Next, we discuss the revert algorithm, specifically bounds on the memory requirements of the same. 

The synthesizer needs to write the reconstructed IQ samples to an output buffer. The sizes of the boxes to be reverted cannot be known in advance. Therefore, in order to bound the size of the output memory in advance, we need to give a reasonable upper bound on the number of boxes to be reverted. The interesting thing about the following analysis is that it would return a bound on the number of boxes that are 'small' in some sense.

Suppose we process a chunk of samples of size $S / 2$. The output of the $N_{\text{ch}}$ channelizer would then have size

$$ N_{\text{ch}} * N_{\text{sl}} = S $$ 

where $N_{\text{sl}}$ is the number of samples in each channel at output.

The algorithm implemented here, first preallocates buffers of size $T_i = 2^i T$ for $1\leq i\leq k$. The dimensions of these buffers in the time and frequency dimensions are of the form $\{T, 2T, 2^2T,\cdots\}$ and $\{F, 2F, 2^2 F, \cdots\}$. Then, for a given box, the tightest buffer that covers the box will be selected to copy the samples of the box from the full channogram with appropriate padding, and apply the synthesis filter for that buffer size.

Now, consider the $j$-th box. Suppose its size is $A_j$ and the time and frequency dimensions of this box are $(t_j, f_j)$. Then, exactly one of the following conditions hold

1. Either $2^k T \leq t_j \leq 2^{k+1} T$ and $2^l F\leq f_j\leq 2^{l+1} F$ for some $k, l$. In this case, we can write that the size of the box $A_j$ satisfies 

$$ A_j \leq T_i \leq 4A_j $$

2. $4A_k \leq T$, where $T$ is the size of the smallest pre-selected buffer.

Boxes satisfying the first condition will be called "large", while boxes satisfying the second condition will be called "small". Thus, processing a large box needs an output memory allocation of size $\frac{T_i}{2}$. 

Whereas, for a small box, an allocation of size $T / 8$ is needed.

Next, large boxes will not be a problem because the memory allocation required is 

$$ \sum_{k| A_k \text{ is a large box}} T_i \leq 4 \sum_{k | A_k \text{ is a large box}} A_k \leq 4S $$

since the boxes are non-overlapping in the channogram.

This proves that large boxes can be processed with an output memory allocation of size at most $2S$, which is twice the size of the original channogram.

For small boxes, we have,

$$ \sum_{k| A_k \text{ is a small box}} T = N_{\text{small}} T$$

Thus, the total output size for the small boxes can be bounded by $\frac{N_{\text{small}} T}{8}$. If we want to bound this by $S$, then, we get the following condition

$$ N_{\text{small}}\leq \frac{8S}{T}$$

This shows that the smaller $T$ is, the number of small boxes the implementation can process with the same fixed output buffer size is greater.

Usage instructions coming soon.



