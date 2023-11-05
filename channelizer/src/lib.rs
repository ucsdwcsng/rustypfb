extern crate offlinepfb_sys;
use num::Complex;
use offlinepfb_sys::{
    chann_create, chann_process, chann_destroy, memory_allocate, memory_allocate_cpu, memory_deallocate,
    memory_deallocate_cpu, transfer, Chann,
};
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

// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
