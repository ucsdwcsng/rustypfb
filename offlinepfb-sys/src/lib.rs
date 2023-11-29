use libm::sinf;
use libm::sqrtf;
use num::Complex;
use std::time::Instant;
/*
 * This is the Rust side declaration of the
 * chann C ABI, which itself is an opaque struct on the
 * C side. The C struct acts as the interface through which the
 * channelizer object defined in CUDA talks to the outside world.
 */
#[repr(C)]
pub struct Chann {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

//
//Rust declarations of C side helper functions that manipulate data on the GPU.
// #[link(name = "cufft", kind = "dylib")]
extern "C" {
    /// C side channelizer constructor. Inputs are filter coefficients, number of prototype filter taps per channel,
    /// number of channels, and number of slices that we want to process per channel.
    pub fn chann_create(
        coeff_arr: *mut Complex<f32>,
        nproto: i32,
        nchann: i32,
        nsl: i32,
    ) -> *mut Chann;

    /// Function that deallocates everything related to a channelizer. Calls the C++ destructor under the hood.
    /// Needed because C style linkage does not allow implicit destructor calls.
    pub fn chann_destroy(channobj: *mut Chann);

    /// The function that does all the heavy lifting under the hood for processing the input data chunks into the
    /// specified number of channels. The output is written to a GPU array and outp points to that array.
    pub fn chann_process(channobj: *mut Chann, inp: *mut f32, outp: *mut Complex<f32>);

    /// Allocated GPU memory for the specified number of float 32s. Generics not allowed
    /// for C style linkage, therefore, this only works for float 32s.
    pub fn memory_allocate_device(inp: i32) -> *mut Complex<f32>;

    /// Deallocates memory on GPU that is pointed to by the input pointer.
    pub fn memory_deallocate_device(inp: *mut Complex<f32>);

    /// CPU memory allocation function. Returns the pointer to the allocated heap memory.
    pub fn memory_allocate_cpu(inp: i32) -> *mut Complex<f32>;

    /// Deallocates CPU memory pointed to by the input.
    pub fn memory_deallocate_cpu(inp: *mut Complex<f32>);

    /// Bessel function computation that used C++17 cmath library. This is used because
    /// I did not find any scientific Rust libraries that can compute these things faster than this.
    pub fn bessel_func(inp: f32) -> f32;

    /// Function that transfers count number of f32s from GPU to CPU, with the input and output pointers specified.
    pub fn transfer(inp: *mut Complex<f32>, outp: *mut Complex<f32>, count: i32);
}
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        use super::{memory_allocate_device, memory_deallocate_device, bessel_func};
        for ind in 0..10 {
            let d = unsafe {memory_allocate_device(100) };
            unsafe { memory_deallocate_device(d) };
        }
    }
}
