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
    fn chann_destroy(channobj: *mut Chann);

    /// The function that does all the heavy lifting under the hood for processing the input data chunks into the
    /// specified number of channels. The output is written to a GPU array and outp points to that array.
    fn chann_process(channobj: *mut Chann, inp: *mut f32, outp: *mut Complex<f32>);

    /// Allocated GPU memory for the specified number of float 32s. Generics not allowed
    /// for C style linkage, therefore, this only works for float 32s.
    fn memory_allocate(inp: i32) -> *mut Complex<f32>;

    /// Deallocates memory on GPU that is pointed to by the input pointer.
    fn memory_deallocate(inp: *mut Complex<f32>);

    /// CPU memory allocation function. Returns the pointer to the allocated heap memory.
    fn memory_allocate_cpu(inp: i32) -> *mut Complex<f32>;

    /// Deallocates CPU memory pointed to by the input.
    fn memory_deallocate_cpu(inp: *mut Complex<f32>);

    /// Bessel function computation that used C++17 cmath library. This is used because
    /// I did not find any scientific Rust libraries that can compute these things faster than this.
    fn bessel_func(inp: f32) -> f32;

    /// Function that transfers count number of f32s from GPU to CPU, with the input and output pointers specified.
    fn transfer(inp: *mut Complex<f32>, outp: *mut Complex<f32>, count: i32);
}

///This struct defines the Rust side interface to a GPU array of complex float 32 numbers.
///The ptr field of this struct will NOT be dereferenced on the Rust side. All manipulations
///of this field will occur in unsafe blocks inside member functions of the DevicePtr struct.
///The functions on the CUDA side which will perform these manipulations
///will be declared to have extern C linkage.
pub struct DevicePtr {
    ptr: *mut Complex<f32>,
    size: i32,
}

impl DevicePtr {
    pub fn new(sz: i32) -> Self {
        // println!("Memory getting allocated in Rust\n");
        Self {
            ptr: unsafe { memory_allocate(sz) },
            size: sz,
        }
    }

    pub fn display(&self, count: i32) {
        unsafe {
            let z = memory_allocate_cpu(count);
            transfer(self.ptr, z, count);
            for ind in 0..(count as usize) {
                println!(
                    "{}, {}\n",
                    (*z.offset(ind as isize)).re,
                    (*z.offset(ind as isize)).im
                );
            }
            memory_deallocate_cpu(z);
        };
    }
}

impl Drop for DevicePtr {
    fn drop(&mut self) {
        // println!("Memory getting deallocated in Rust");
        unsafe { memory_deallocate(self.ptr) };
    }
}

pub struct RustChannelizer {
    opaque_chann: *mut Chann,
}

impl RustChannelizer {
    pub fn new(inp: &[f32], proto_taps: i32, channels: i32, slices: i32) -> Self {
        // println!("Chanelizer getting created\n");
        let mut complex_coeff_array: Vec<Complex<f32>> =
            inp.iter().map(|x| Complex::new(*x, 0.0)).collect();
        Self {
            opaque_chann: unsafe {
                chann_create(
                    complex_coeff_array.as_mut_ptr(),
                    proto_taps,
                    channels,
                    slices,
                )
            },
        }
    }

    pub fn process(&mut self, inp: &mut [f32], output: &mut DevicePtr) {
        unsafe { chann_process(self.opaque_chann, inp.as_mut_ptr(), output.ptr) }
    }
}

impl Drop for RustChannelizer {
    fn drop(&mut self) {
        // println!("Channelizer destroyed!");
        unsafe { chann_destroy(self.opaque_chann) };
    }
}

fn main() {
    const NCH: i32 = 1024;
    const NSLICE: i32 = 1024 * 128 * 2;
    const NPROTO: i32 = 100;
    const NSAMPLES: i32 = 100000000;
    let kbeta: f32 = 9.6;
    let mut coeff_array: Vec<f32> = Vec::new();
    for ind in 0..((NCH * NPROTO) as usize) {
        let darg = ((2 * ind) as f32) / ((NCH * NPROTO) as f32) - 1.0;
        let carg = kbeta * sqrtf(1.0 - darg * darg);
        coeff_array.push(unsafe { bessel_func(carg) / bessel_func(kbeta) });
    }

    let mut chann_obj: RustChannelizer =
        RustChannelizer::new(coeff_array.as_mut_slice(), NPROTO, NCH, NSLICE);
    let mut output_buffer: DevicePtr = DevicePtr::new(NCH * NSLICE);
    let mut input_vec: Vec<f32> = Vec::new();
    for ind in 0..((2 * NSAMPLES) as usize) {
        if ind % 2 == 0 {
            input_vec.push(sinf(ind as f32));
        } else {
            input_vec.push(sinf(ind as f32) / (ind as f32));
        }
    }
    let mut m: i32 = 0;
    let mut tot: u128 = 0;
    loop {
        m += 1;
        let n = Instant::now();
        chann_obj.process(&mut input_vec, &mut output_buffer);
        let w = n.elapsed().as_millis();
        tot += w;
        // output_buffer.display(10);
        // println!("In loop number {}", m);
        if m == 100 {
            break;
        }
    }
    println!("Process calls take {} milliseconds on average", tot / 100);
}
