Consider the following operation on the sequence $x[n]$:\
![Signal Processing Operation that we want to do.](https://github.com/ucsdwcsng/rust_channelizer/blob/main/docs/channelizer.png)


1. The frequency band of the input centered at $\theta$ is first downconverted to baseband (this is signal speak for shifting the entire frequency spectrum of $x[n]$ to the left by $\theta$). 

2. Then, the downconverted input is passed through a filter $H(Z)$ a notation which simply represents
$$H(Z) = \sum_{n=0}^\infty h_n z^{-n}.$$ 

In this notation, we multiply the z-transform of $x[n]$ by $H(Z)$ which in time domain simply implies convolution of $x[n]$ by the filter coefficients $h_n$.

3. Finally, the output of the filter is downsampled/subsampled by a factor of $M$. This is signalspeak for only keeping every $M$-th element in the input. Note that the filter is needed, otherwise, a naive subsampling causes aliasing effects in the spectral domain.

This downconvert, filter and then subsample pattern is routinely performed in the case of analyzing isolated frequency bands of the input signals. Therefore, it is of utmost improtance that this be done quickly. We explore the technique of polyphase decomposition of $H(Z)$ to do this.

We first note the following:
$$\sum_{k=-\infty}^\infty x[k]e^{-j\theta k} h_{n-k} = e^{-j\theta n}\sum_{k}x[k] h_{n-k}e^{j\theta(n-k)}$$

This is as if, the filter coefficients $h_n$ have been replaced by $h_n e^{j\theta n}$. This implies the following rearrangement of the block diagram above\
![Swap filtering and downconversion](https://github.com/ucsdwcsng/rust_channelizer/blob/main/docs/channelizer_2.png)

A result that is known as the equivalence theorem. We can then shift the multiplication by the complex sinusoid across the downsampler, and we get this:
![Swap downsampling and downconversion](https://github.com/ucsdwcsng/rust_channelizer/blob/main/docs/channelizer_3.png).

The last multiplication can be removed with the assumption that 

$$ M\theta = 2\pi l$$ 

so that we are only interested in center frequencies that are integer multiples of the downsampling rate. Thus, we need to analyze the following:\
![Removing downconversion](https://github.com/ucsdwcsng/rust_channelizer/blob/main/docs/channelizer_4.png).

We next note that as written here, we are computing the convolution of the filter with the input at the full rate, i.e., we are computing all the values of the output, but discarding every $M$-th value. This is wasteful, and the Polyphase Channelizer algorithm is designed to remedy this.

Instead of thinking about delays and polyphase components etc., let us directly compute every $M$-th output of the filtering operation, and see if we observe any patterns.

The Z-transforms of the input $x[n]$ (which we assume starts at time instant $n=1$) is simply

$$X(z) = x_1 z^{-1} + x_2 z^{-2}+\cdots$$

while the filter Z-transform is (note that we are taking downconversion into account from the get-go, in the following, note $\xi = e^{j\theta}$)

$$H(Z e^{-j\theta}) = h_0 + h_1 \xi z^{-1} + h_2\xi^2 z^{-2}+\cdots$$

Now, simply multiply $H(Z\xi^{-1})$ with $X(Z)$ to obtain

$$F(Z, \xi) = \biggl(x_1 Z^{-1} + x_2 Z^{-2}+\cdots\biggr)\cdot\biggl(h_0 + h_1 \xi Z^{-1} + h_2\xi^2 Z^{-2}+\cdots\biggr)$$

Since we are interested in every $M$-th output of this product, it makes sense to group together all those filter coefficients whose indices leave the same remainder when divided by $M$. Thus, we decompose the product $F(Z, \xi)$ as

$$F(Z, \xi) = F_0(Z, \xi) + F_1(Z, \xi) + \cdots + F_{M-1}(Z, \xi)$$

Here, the notation $F_j(Z, \xi)$ indicates the portion of $F$ that depends on filter coefficients $h_k$ such that $k=pM + j$ for some $p$. Note futher that, at the output due to the downsamplig, we are only interested in terms in the above that involve $Z^{-lM}$ for $l$ an integer. We also note in passing $\xi^M = 1$. 

Consider $F_0(Z, \xi)$. This will only involve the coefficients $h_0, h_M, h_{2M}, \cdots$. Collecting all these coefficients in $F(Z, \xi)$, and only looking at the coefficients of $Z^{-lM}$ for some $l$ an integer, we get

$$F_0(Z, \xi) = (h_0 x_M) Z^{-M} + (h_M x_M + h_0 x_{2M})Z^{-2M}$$ 

$$ + (h_0 x_{3M} + h_M x_{2M} + h_{2M} x_{M})z^{-3M} + \cdots = G_0(Z)$$

Next, look at coefficients of $Z^{-lM}$ for some $l$ an integer in $F_1(Z, \xi)$,

$$F_1(Z, \xi) = \xi\biggl(h_1 x_{M-1} Z^{-M} + (h_{M+1} x_{M-1} + h_1 x_{2M-1})Z^{-2M}\biggr)$$ 

$$+\xi\biggl((h_1 x_{3M-1} + h_{M+1} x_{2M-1} + h_{2M+1} x_{M-1})z^{-3M} + \cdots\biggr) = \xi G_1(Z) $$

Analyze $F_2(Z, \xi)$ next,

$$F_2(Z, \xi) = \xi^2\biggl(h_2 x_{M-2} Z^{-M} + (h_{M+2} x_{M-2} + h_2 x_{2M-2})Z^{-2M}\biggr)$$ 

$$+\xi^2\biggl((h_2 x_{3M-2} + h_{M+2} x_{2M-2} + h_{2M+2} x_{M-2})z^{-3M} + \cdots\biggr) = \xi^2 G_2(Z)$$

Observe :

1. The $j$-th component $F_j$ depends on $\xi^j$ solely as the factor $\xi^j$ multiplied by a quantity $G_j(Z)$ that does not depend on $\xi$.

2. To compute $G_j(Z)$, simply take the input samples $x_{M-j}, x_{2M-j}, x_{3M-j}, \cdots$, and perform the convolution of these samples with the filter coefficients $h_j, h_{M+j}, h_{2M+j}\cdots$. 

3. Note that before these observations, we were computing convolutions between filter and input at full rate. But now, we only need to perform one convolution computation every $M$ samples of the input. This represents the gains we have made with this observation.

4. Thus, to obtain the output at a given value of the center frequency, set by $\xi = e^{j\theta}$, simply compute $G_j(Z)$ as above, and then
compute

$$\sum_{j=0}^{M-1}\xi^j G_j(Z)$$

4. This represents the output at a given value of the center frequency. What if we want all of them? This structure suggests that the answer is to compute $G_j(Z)$ for each $j$ and then simply take the IFFT. The output of the IFFT would contain the filterd+downconverted+downsampled signal at all possible center frequencies.


Offline Channelizer :
This is the computation one needs to perform to get disjoint channels from the input array $x[n]$ given filter coefficients $h_n$. One firsts arranges the input array $x[n]$ in the layout in a buffer array

$$
X = \begin{pmatrix}
x_M  & x_{2M} & x_{3M} & \cdots \\
x_{M-1} & x_{2M-1} & x_{3M-1} &\cdots \\
x_{M-2} & x_{2M-2} & x_{3M-2} &\cdots \\
           &\vdots \\
x_1  & x_{M+1}& x_{2M+1} & \cdots
\end{pmatrix}
$$

This step will involve copy of the input into the layout given above. Successive input elements are mapped onto non-contiguous portions of the buffer array in the above layout, and therefore, this step is a time consuming one.

Filter coefficients will be stored in the polyphase layout

$$
H = \begin{pmatrix}
h_0 & h_{M} & h_{2M} & \cdots \\
h_1 & h_{M+1} & h_{2M+1} &\cdots \\
h_2 & h_{M+2} & h_{2M+2} &\cdots \\
          &\vdots \\
h_{M-1} & h_{2M-1} & h_{3M-1} &\cdots
\end{pmatrix}
$$

The output is computed simply as 

$$ \text{Column-wise IFFT}\biggl(\text{Row-wise Convolve}\biggl(X, H\biggr)\biggr)$$

For the off-line version, we use FFT based convolution algorithms as they would be faster than multiplcation based convolution.

Online Channelizer:

One can also construct a streaming channelizer by a simple modification of the algorithm above. By a streaming channelizer, we mean that the output in each channel is computed one sample at a time.

Since there are $M$ channels, the channelizer has to wait for $M$ input samples to compute one output sample in each channel. This is simply a consequence of the downsampling operation. (One can infact start computing new outputs given $M/2$ samples actually, but we don't do that micro-optimization here).

Call the maximum number of (non-zero) filter coefficients per channel as the number of taps per channel (denoted as $T$). Then, one can construct a streaming channelizer by maintaining $M$ circular buffers of size $T$ each, one for each polyphase component of the filter. 

The first sample in each batch of $M$ new samples is sent to the last polyphase component, the second to the second-last polyphase component, and so on. The insertion of these new samples updates the buffers. Then, compute the convolution of each buffer with the corresponding polyphase filter component by simple addition and multiplication. This would be a number for each channel. Collect them in a vector and then take the IFFT of the resulting vector. That would be the updated output from the channelizer given the $M$ inputs.
