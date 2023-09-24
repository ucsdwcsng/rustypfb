use num::{Complex, complex};
/*
 * This is the Rust side declaration of the
 * chann C ABI, which itself is an opaque struct on the 
 * C side. The C struct acts as the interface through which the 
 * channelizer object defined in CUDA talks to the outside world.
 */
#[repr(C)]
pub struct chann {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/*
 * Rust declarations of C side helper functions that manipulate data on the GPU.
 * 
 * 1. chann_create creates a channelizer object on the CUDA side, 
 *    with various memory allocations for all the internal GPU buffers.
 * 
 * 2. chann_destroy frees the memory occupied by the channelizer object.
 * 
 * 3. chann_process performs the forward channelization process on 
 * interleaved floats.
 * 
 * 4. memory_allocate allocates memory on a GPU buffer and returns a pointer to the buffer.
 */
extern "C" {
    fn chann_create(coeff_arr: *const Complex<f32>) -> *mut chann;
    fn chann_destroy(channobj: *mut chann);
    fn chann_process(channobj: *mut chann, inp: *mut f32, outp: *mut Complex<f32>);
    fn memory_allocate(inp: i32) -> *mut Complex<f32>;
}

/*
 * This struct defines the Rust side interface to a GPU array of complex float 32 numbers.
 * The ptr field of this struct will NOT be dereferenced on the Rust side. All manipulations 
 * of this field will occur in unsafe blocks inside member functions of the DevicePtr struct.
 * The functions on the CUDA side which will perform these manipulations
 * will be declared to have extern C linkage.
 */

 pub struct DevicePtr {
    ptr: *mut Complex<f32>,
    size: i32,
}

impl DevicePtr {
    pub fn new(sz: i32) -> Self {
        Self {
            ptr : unsafe { memory_allocate(sz) },
            size: sz,
        }
    }
}

pub struct RustChannelizer {
    OpaqueChann: *mut chann,
}

impl RustChannelizer{
    pub fn new(inp: &[f32], size: i32) -> Self 
    {
        let complex_coeff_array: Vec<Complex<f32>> =
        inp.iter().map(|x| Complex::new(*x, 0.0)).collect();
        Self {
            OpaqueChann: unsafe{chann_create(complex_coeff_array.as_ptr())},
        }

    }

    pub fn process(&mut self, inp: &mut [f32], output: &mut DevicePtr)
    {
        unsafe{chann_process(self.OpaqueChann, inp.as_mut_ptr(), output.ptr)}
    }
}

impl Drop for RustChannelizer {
    fn drop(&mut self)
    {
        unsafe{chann_destroy(self.OpaqueChann)};
    }
}

fn main() {
    println!("Hello World!");
}
