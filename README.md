This is an implementation of the Polyphase Filter Bank (PFB) in the Rust programming language. Two versions of the Channelizer have been implemented here. 

One is an offline version that channelizes a chunk of fixed size into a given number of channels. This version has been implemented on the GPU. The Rust code interfaces with CUDA C++ for performance critical aspects of the computation. 

The other is an online version that returns output in a streaming fashion, with the smallest possible latency. Thus, with 1024 channels, this channelizer returns an output in each channel after waiting for every 512 samples of input. 