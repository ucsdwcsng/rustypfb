extern crate offlinepfb_sys;
use bytemuck::Pod;
use libm::sinf;
use libm::sqrtf;
use num::Complex;
use offlinepfb_sys::{
    bessel_func, chann_create, chann_destroy, chann_process, memory_allocate_cpu,
    memory_allocate_device, memory_deallocate_cpu, memory_deallocate_device, transfer, Chann,
};
use std::io::Read;
/// This struct defines the Rust side interface to a GPU array of complex float 32 numbers.
/// The ptr field of this struct will NOT be dereferenced on the Rust side. All manipulations
/// of this field will occur in unsafe blocks inside member functions of the DevicePtr struct.
/// The functions on the CUDA side which will perform these manipulations
/// will be declared to have extern C linkage.
pub struct DevicePtr {
    ptr: *mut Complex<f32>,
    size: i32,
}

pub fn sinc(inp: f32) -> f32 {
    if inp == 0.0 {
        0.0
    } else {
        inp.sin() / inp
    }
}
impl DevicePtr {
    pub fn new(sz: i32) -> Self {
        // println!("Memory getting allocated in Rust\n");
        Self {
            ptr: unsafe { memory_allocate_device(sz) },
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
        unsafe { memory_deallocate_device(self.ptr) };
    }
}

pub struct ChunkChannelizer {
    opaque_chann: *mut Chann,
}

impl ChunkChannelizer {
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

impl Drop for ChunkChannelizer {
    fn drop(&mut self) {
        // println!("Channelizer destroyed!");
        unsafe { chann_destroy(self.opaque_chann) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use std::mem::{self, align_of_val};
    #[test]
    fn it_works() {
        // Setup the Channelizer
        let nch = 1024;
        let ntaps = 128;
        let nslice = 2 * 1024 * 128;
        let float_taps = (-ntaps / 2) as f32;
        let chann_float = nch as f32;
        let chann_proto = ntaps as f32;
        let kbeta = 10 as f32;
        let mut filter: Vec<f32> = (0..nch * ntaps)
            .map(|x| {
                let y = x as f32;
                let arg = float_taps + (y + 1.0) / chann_float;
                let darg = (2.0 * y) / (chann_float * chann_proto) - 1.0;
                let carg = kbeta * (1.0 - darg * darg).sqrt();
                (unsafe { bessel_func(carg) / bessel_func(kbeta) }) * sinc(arg)
            })
            .collect();
        let mut chann_obj = ChunkChannelizer::new(filter.as_mut_slice(), ntaps, nch, nslice);

        // Read data from file
        let mut file = std::fs::File::open("../busyBand/DSSS.32cf").unwrap();
        let mut samples_bytes = Vec::new();
        let _ = file.read_to_end(&mut samples_bytes);
        let samples: &[f32] = bytemuck::cast_slice(&samples_bytes);
        // Commenting this line out causes segmentation fault
        let samples_vec: Vec<f32> = samples.to_vec();

        // Copy data to input buffer of appropriate size
        const NSAMPLES: i32 = 100000000;
        let mut input_vec = vec![0.0 as f32; (2 * NSAMPLES) as usize];
        input_vec[..samples.len()].clone_from_slice(samples);
        // Setup the output buffer
        let mut output_buffer: DevicePtr = DevicePtr::new(nch * nslice);

        // Process
        chann_obj.process(&mut input_vec, &mut output_buffer);

        // Display
        output_buffer.display(100);
    }
}
