use circular_buffer::CircularBuffer;
use libm::cos;
use libm::erfc;
use libm::log;
use libm::sin;
use libm::sqrt;
use num::Complex;
use rayon::prelude::*;
use rustfft::Fft;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/*
 * Number of taps per channel. Have to specify as a global constant
 * otherwise CircularBuffer will complain.
 */
const taps: usize = 100;
/*
 Streaming version of the polyphase channelizer.
 Updates outputs one slice across channels at a time.
 This is maximally decimating.
*/
pub struct Channelizer {
    nchannels: usize, // This is the number of channels.

    // This is the number of taps per channel.

    // Filter coefficients in polyphase form
    coeff: Vec<Vec<Complex<f64>>>,

    // // Buffer to hold polyphase products
    // buffer: Vec<Complex<f64>>>,

    // Plan for IFFT
    plan: Box<dyn Fft<f64>>,

    // Collection of CircularBuffers for each channel. Each buffer has capacity = taps.
    internal_buffers: Vec<CircularBuffer<taps, Complex<f64>>>,

    // pre_out_buffer
    pre_out_buffer: Vec<Complex<f64>>,

    // Output Buffer : This represents the current one slice output across channels
    out_buffer: Vec<Complex<f64>>,

    // Scratch buffer for FFT
    scratch_buffer: Vec<Complex<f64>>,
}

impl Channelizer {
    pub fn process(&mut self, sample_arr: &[Complex<f64>]) {
        self.internal_buffers
            .par_iter_mut()
            .enumerate()
            .for_each(|(ind, item)| item.push_front(sample_arr[self.nchannels - ind]));

        (self.pre_out_buffer)
            .par_iter_mut()
            .enumerate()
            .for_each(|(ind, item)| (*item) = buffer_process(&self.internal_buffers, &self.coeff, ind));

    }
}

pub fn buffer_process(
    lhs: &Vec<CircularBuffer<taps, Complex<f64>>>,
    rhs: &Vec<Vec<Complex<f64>>>,
    id: usize,
) -> Complex<f64> {
    let mut sum = Complex { re: 0.0, im: 0.0 };
    for (ind, item) in lhs[id].iter().enumerate() {
        sum += (*item) * (rhs[id][ind]);
    }
    sum
}
