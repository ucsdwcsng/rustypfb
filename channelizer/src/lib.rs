extern crate offlinepfb_sys;
extern crate rustdevice;
use bytemuck::Pod;
use libm::sinf;
use libm::sqrtf;
use num::Complex;
use offlinepfb_sys::{
    chann_create, chann_destroy, chann_process, Chann,
    chann_set_revert_filter, chann_revert,
};
use rustdevice::{DevicePtr, compute_bessel};
use std::io::Read;
use std::f32::consts::PI;
/// This struct defines the Rust side interface to a GPU array of complex float 32 numbers.
/// The ptr field of this struct will NOT be dereferenced on the Rust side. All manipulations
/// of this field will occur in unsafe blocks inside member functions of the DevicePtr struct.
/// The functions on the CUDA side which will perform these manipulations
/// will be declared to have extern C linkage.
// pub struct DevicePtr {
//     pub ptr: *mut Complex<f32>,
//     pub size: i32,
// }

pub fn sinc(inp: f32) -> f32 {
    if inp == 0.0 {
        1.0
    } else {
        (PI*inp).sin() / (PI*inp)
    }
}
// impl DevicePtr {
//     pub fn new(sz: i32) -> Self {
//         // println!("Memory getting allocated in Rust\n");
//         Self {
//             ptr: unsafe { memory_allocate_device(sz) },
//             size: sz,
//         }
//     }

//     pub fn display(&self, count: i32) {
//         unsafe {
//             let z = memory_allocate_cpu(count);
//             transfer(self.ptr, z, count);
//             for ind in 0..(count as usize) {
//                 println!(
//                     "{}, {}\n",
//                     (*z.offset(ind as isize)).re,
//                     (*z.offset(ind as isize)).im
//                 );
//             }
//             memory_deallocate_cpu(z);
//         };
//     }
// }

// impl Drop for DevicePtr {
//     fn drop(&mut self) {
//         // println!("Memory getting deallocated in Rust");
//         unsafe { memory_deallocate_device(self.ptr) };
//     }
// }

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

    pub fn process(&mut self, inp: &mut [f32], output: &mut DevicePtr<Complex<f32>>) {
        unsafe { chann_process(self.opaque_chann, inp.as_mut_ptr(), output.ptr) }
    }

    pub fn revert(&mut self, inp: &mut DevicePtr<Complex<f32>>, output: &mut DevicePtr<Complex<f32>>)
    {
        unsafe{chann_revert(self.opaque_chann, inp.ptr, output.ptr)}
    }

    pub fn set_revert_filter(&mut self, inp: &[f32])
    {
        let mut complex_coeff_array: Vec<Complex<f32>> =
            inp.iter().map(|x| Complex::new(*x, 0.0)).collect();
        unsafe{ chann_set_revert_filter(self.opaque_chann, complex_coeff_array.as_mut_ptr())}
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
    use num::Zero;

    use super::*;
    use std::io::Write;
    use std::time::Instant;
    use std::mem::{self, align_of_val};
    #[test]
    fn correctness_visual_test() {
        // Setup the Channelizer
        let nch = 1024;
        let ntaps = 128;
        let nslice = 65536;
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
                (unsafe { compute_bessel(carg) / compute_bessel(kbeta) }) * sinc(arg)
            })
            .collect();
        let mut chann_obj = ChunkChannelizer::new(filter.as_mut_slice(), ntaps, nch, nslice);

        // Setup the output buffer
        let mut output_buffer = DevicePtr::<Complex<f32>>::new(nch * nslice);
        
        // Setup the CPU output buffer
        let mut output_cpu = vec![Complex::<f32>::zero(); (nch*nslice) as usize];
        
        // Setup the input vector
        let mut input_vec = vec![0.0 as f32; (nch*nslice) as usize];
        
        /*
         * DSSS test
         */
        let mut dsss_file = std::fs::File::open("../busyBand/DSSS.32cf").unwrap();
        let mut dsss_samples_bytes = Vec::new();
        let _ = dsss_file.read_to_end(&mut dsss_samples_bytes);
        let dsss_samples: &[f32] = bytemuck::cast_slice(&dsss_samples_bytes);
        // println!("{}", samples.len());
        // Copy onto input
        input_vec[..dsss_samples.len()].clone_from_slice(dsss_samples);

        // Process
        chann_obj.process(&mut input_vec, &mut output_buffer);
        // output_buffer.display(10);
        // Transfer
        // unsafe{transfer(output_buffer.ptr, output_cpu.as_mut_ptr(), nch*nslice)};
        output_buffer.dump(&mut output_cpu);

        let mut dsss_file = std::fs::File::create("../dsss_chann_output.32cf").unwrap();

        let dsss_outp_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut output_cpu);

        let _ = dsss_file.write_all(dsss_outp_slice);


        // Reset input
        input_vec.iter_mut().for_each(|x| *x = 0.0);


        /*
         * LPI combined
         */
        let mut lpi_file = std::fs::File::open("../busyBand/lpi_combined.32cf").unwrap();
        let mut lpi_samples_bytes = Vec::new();
        let _ = lpi_file.read_to_end(&mut lpi_samples_bytes);
        let lpi_samples: &[f32] = bytemuck::cast_slice(&lpi_samples_bytes);
        // println!("{}", samples_.len());
        // Copy onto input
        // let mut input_vec_ = vec![0.0 as f32; (nch*nslice) as usize];
        input_vec[..lpi_samples.len()].clone_from_slice(lpi_samples);

        // // Setup the output buffer
        // let mut output_buffer_: DevicePtr = DevicePtr::new(nch * nslice);

        // Process
        chann_obj.process(&mut input_vec, &mut output_buffer);

        // let mut output_cpu_ = vec![Complex::<f32>::zero(); (nch*nslice) as usize];

        // Transfer
        // unsafe{transfer(output_buffer.ptr, output_cpu.as_mut_ptr(), nch*nslice)};
        output_buffer.dump(&mut output_cpu);

        let mut lpi_file = std::fs::File::create("../lpi_chann_output.32cf").unwrap();

        let lpi_outp_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut output_cpu);

        let _ = lpi_file.write_all(lpi_outp_slice);


    }
}
