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

The Z-transforms of the input $x[n]$ is simply

$$X(z) = x_0 + x_1 z^{-1} + x_2 z^{-2}+\cdots$$

while the filter Z-transform is (note that we are taking downconversion into account from the get-go, in the following, note $\xi = e^{-j\theta}$)

$$H(Z e^{-j\theta}) = h_0 + h_1 \xi z^{-1} + h_2\xi^2 z^{-2}+\cdots$$

Now, simply multiply $H(Z\xi)$ with $X(Z)$ to obtain

$$F(Z, \xi) = \biggl(x_0 + x_1 z^{-1} + x_2 z^{-2}+\cdots\biggr)\cdot\biggl(h_0 + h_1 \xi z^{-1} + h_2\xi^2 z^{-2}+\cdots\biggr)$$

Since we are interested in every $M$-th output of this product, it makes sense to group together all those filter coefficients whose indices leave the same remainder when divided by $M$. Thus, we decompose the product $F(Z, \xi)$ as

$$F(z, \xi) = F_0(z, \xi) + F_1(z, \xi) + \cdots + F_{M-1}(z, \xi)$$.

Here, the notation $F_j(z, \xi)$ indicates the portion of $F$ that depends on filter coefficients $h_k$ such that $k=pM + j$ for some $p$. We also note in passing $\xi^M = 1$. 

Consider the coefficients $h_0, h_M, h_{2M}, \cdots$. These enter the above product in 