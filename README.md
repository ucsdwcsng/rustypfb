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

The synthesizer needs to write the reconstructed IQ samples to an output buffer. The sizes of the boxes to be reverted cannot be known in advance. Therefore, in order to bound the size od the output memory in advance, we need to give a reasonable upper bound on the number of boxes to be reverted. The interesting thing about the following analysis is that it would return a bound on the number of boxes that are 'small' in some sense.

Suppose we process a chunk of samples of size $S / 2$. The output of the $N_{\text{ch}}$ channelizer would then have size

$$ N_{\text{ch}} * N_{\text{sl}} = S $$ 

where $N_{\text{sl}}$ is the number of samples in each channel at output.

The algorithm implemented here, first preselects buffers of size $T_i$ for $1\leq i\leq k$. These buffers will copy the samples of the to-be-reverted box from the full channogram with appropriate padding, and apply the synthesis filter for that buffer size.

Now, consider the $j$-th box. Suppose its size is $A_k$. Then, exactly one of the following two conditions hold

1. Either $A_k \leq T_i \leq 2 A_k$ for some $i$, or,
2. $2A_k \leq T$, where $T$ is the size of the smallest pre-selected buffer.

Boxes satisfying the first condition will be called "large", while boxes satisfying the second condition will be called "small". Thus, processing a large box needs an output memory allocation of size $\frac{T_i}{2}$. 

Whereas, for a small box, an allocation of size $T / 4$ is needed.

Next, large boxes will not be a problem because the memory allocation required is 

$$ \sum_{k| A_k \text{ is a large box}} T_i \leq 2 \sum_{k | A_k \text{ is a large box}} A_k \leq 2S $$

since the boxes are non-overlapping in the channogram.

This proves that large boxes can be processed with an output memory allocation of size at most $S$, which is the size of the original channogram.

For small boxes, we have,

$$ \sum_{k| A_k \text{ is a small box}} T = N_{\text{small}} T$$.

Thus, the total output size for the small boxes can be bounded by $\frac{N_{\text{small}} T}{4}$. If we want to bound this by $S$, then, we get the following condition

$$ N_{\text{small}}\leq \frac{4S}{T}$$.

This shows that the smaller $T$ is, the larger the number of small boxes the implementation can process with the same fixed output buffer size.

Usage instructions coming soon.



