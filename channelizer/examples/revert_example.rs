use analyzer::SSCAWrapper;
pub use channelizer::{sinc, ChunkChannelizer};
use num::{Complex, Zero};
use num_complex::Complex32;
pub use rustdevice::{compute_bessel, DevicePtr};
use std::io::{Read, Write};
fn main() {
    // Set up the Channelizer
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
            (compute_bessel(carg) / compute_bessel(kbeta)) * sinc(arg)
        })
        .collect();

    let mut chann_obj = ChunkChannelizer::new(filter.as_mut_slice(), ntaps, nch, nslice);

    let mut revert_filter: Vec<f32> = (0..nch * ntaps)
        .map(|x| {
            let y = x as f32;
            let arg = float_taps + (y + 1.0) / chann_float;
            let darg = (2.0 * y) / (chann_float * chann_proto) - 1.0;
            let carg = kbeta * (1.0 - darg * darg).sqrt();
            (compute_bessel(carg) / compute_bessel(kbeta)) * sinc(2.0 * arg)
        })
        .collect();

    chann_obj.set_revert_filter(&revert_filter);

    // Setup the output buffer
    let mut channelized_output_buffer = DevicePtr::<Complex<f32>>::new(nch * nslice);

    // Setup the revert output buffer
    let mut revert_output_buffer = DevicePtr::<Complex<f32>>::new(nch * nslice / 2);

    // Setup the CPU output buffer
    let mut revert_output_cpu = vec![Complex::<f32>::zero(); (nch * nslice / 2) as usize];

    // // let mut revert_output_buffer_cpu = vec![Complex::<f32>::zero(); (nch*nslice) / 2 as usize];

    // Setup the input vector
    let mut input_vec = vec![0.0 as f32; (nch * nslice) as usize];
    let mut input_vec_complex = vec![Complex::zero() as Complex<f32>; (nch * nslice / 2) as usize];

    /*
     * DSSS test
     */
    let mut dsss_file = std::fs::File::open("./busyBand/DSSS.32cf").unwrap();
    // // println!("{}", std::path::Path{"../busyBand/DSSS.32cf"});
    let mut dsss_samples_bytes = Vec::new();
    let _ = dsss_file.read_to_end(&mut dsss_samples_bytes);
    let dsss_samples: &[f32] = bytemuck::cast_slice(&dsss_samples_bytes);
    let dsss_samples_complex: &[Complex<f32>] = bytemuck::cast_slice(&dsss_samples_bytes);

    // // println!("{}", samples.len());
    // // Copy onto input
    input_vec[..dsss_samples.len()].clone_from_slice(dsss_samples);
    input_vec_complex[..dsss_samples_complex.len()].clone_from_slice(dsss_samples_complex);

    // // Process
    chann_obj.process(&mut input_vec, &mut channelized_output_buffer);

    // // Revert
    chann_obj.revert(&mut channelized_output_buffer, &mut revert_output_buffer);

    // // // Transfer
    revert_output_buffer.dump(&mut revert_output_cpu);

    // // let size_val = 133120 * 8;
    // // let mut ssca_obj = SSCAWrapper::new(size_val);

    // // let outp_size = ssca_obj.get_output_size();
    // // let mut original_sum_buffer = vec![0.0 as f32; outp_size as usize];
    // // let mut original_max_buffer = vec![0.0 as f32; outp_size as usize];

    // // let mut reverted_max_buffer = vec![0.0 as f32; outp_size as usize];
    // // let mut reverted_sum_buffer = vec![0.0 as f32; outp_size as usize];

    // // ssca_obj.process(
    // //     &mut revert_output_cpu,
    // //     false,
    // //     &mut reverted_sum_buffer,
    // //     &mut reverted_max_buffer,
    // // );
    // // ssca_obj.process(
    // //     &mut input_vec_complex,
    // //     false,
    // //     &mut original_sum_buffer,
    // //     &mut original_max_buffer,
    // // );

    let mut dsss_file = std::fs::File::create("dsss_chann_reverted_output.32cf").unwrap();

    let dsss_outp_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut revert_output_cpu);

    let _ = dsss_file.write_all(dsss_outp_slice);

    // // // Reset input
    input_vec.iter_mut().for_each(|x| *x = 0.0);
    input_vec_complex.iter_mut().for_each(|x| *x = Complex32::zero());

     /*
      * LPI combined
      */
    let mut lpi_file = std::fs::File::open("./busyBand/lpi_combined.32cf").unwrap();
    let mut lpi_samples_bytes = Vec::new();
    let _ = lpi_file.read_to_end(&mut lpi_samples_bytes);
    let lpi_samples: &[f32] = bytemuck::cast_slice(&lpi_samples_bytes);
    let lpi_samples_complex: &[Complex<f32>] = bytemuck::cast_slice(&lpi_samples_bytes);
    // println!("{}", samples_.len());
    // Copy onto input
    // let mut input_vec_ = vec![0.0 as f32; (nch*nslice) as usize];
    input_vec[..lpi_samples.len()].clone_from_slice(lpi_samples);
    input_vec_complex[..lpi_samples_complex.len()].clone_from_slice(lpi_samples_complex);

    // // Setup the output buffer
    // let mut output_buffer_: DevicePtr = DevicePtr::new(nch * nslice);

    // Process
    chann_obj.process(&mut input_vec, &mut channelized_output_buffer);

    // let mut output_cpu_ = vec![Complex::<f32>::zero(); (nch*nslice) as usize];

    // Transfer
    // unsafe{transfer(output_buffer.ptr, output_cpu.as_mut_ptr(), nch*nslice)};
    chann_obj.revert(&mut channelized_output_buffer, &mut revert_output_buffer);

    revert_output_buffer.dump(&mut revert_output_cpu);

    let mut lpi_file = std::fs::File::create("./lpi_chann_reverted_output.32cf").unwrap();

    let lpi_outp_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut revert_output_cpu);

    let _ = lpi_file.write_all(lpi_outp_slice);
}
