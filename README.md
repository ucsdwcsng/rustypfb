This is an implementation of the Polyphase Filter Bank (PFB) in the Rust programming language. Two versions of the Channelizer have been implemented here. 

One is an offline version that channelizes a chunk of fixed size into a given number of channels. This version has been implemented on the GPU. The Rust code interfaces with CUDA C++ for performance critical aspects of the computation. 

The other is an online version that returns output in a streaming fashion, with the smallest possible latency. For example, with 1024 channels, this channelizer returns an output in each channel after waiting for every 512 samples of input. It internally maintains circular buffers required for maintenance of state from one slice to next in each channel.

The folder structure of this repo is as follows :

1. The offlinepfb folder contains the native C++ code that performs the bulk of the offline channelizer computation.
2. The offlinepfb-sys folder contains the unsafe Rust bindings to the functions and methods that have been defined in the previous point. This is a sys-crate (that has nevertheless not yet been published to crates.io).
3. The channelizer folder is the Rust module that defines the higher level Rust bindings to the sys crate above. 
4. The streaming_channelizer folder is a simple Rust module that contains the implementations required for the online channelizer.

For a typical user of this repo, these are the steps that need to be followed for setting things up if you want to use the offline channelizer :
1. Clone the repo.
2. In the Cargo.toml of your project, add the relative path to the `channelizer` folder under the dependencies key.
3. You can now use the exposed functions in this module from your project.

In case the online version is needed,  
1. Clone the repo
2. In the Cargo.toml of your project, add the relative path to the `streaming_channelizer` folder under the dependencies key.
3. You can now use the exposed functions in this module from your project.

