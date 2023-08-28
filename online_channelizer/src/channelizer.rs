use circular_buffer::CircularBuffer;
use libm::cos;
use libm::erfc;
use libm::log;
use libm::sin;
use libm::sqrt;
use num::Complex;
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

    // State : Last channel to which a sample was added. Goes from 0 to ncannel - 1
    state: usize,

    // pre_out_buffer
    pre_out_buffer: Vec<Complex<f64>>,

    // Output Buffer : This represents the current one slice output across channels
    out_buffer: Vec<Complex<f64>>,

    // Scratch buffer for FFT
    scratch_buffer: Vec<Complex<f64>>,
}

impl Channelizer {
    pub fn buffer_process(&self, id: usize) -> Complex<f64> {
        let mut sum = Complex { re: 0.0, im: 0.0 };
        for (ind, item) in self.internal_buffers[id].iter().enumerate() {
            sum += (*item) * (self.coeff[id][ind]);
        }
        sum
    }

    pub fn process(&mut self, sample: Complex<f64>) {
        self.state = (self.state + 1) % (self.nchannels);
        self.internal_buffers[self.nchannels - self.state].push_front(sample);
        self.pre_out_buffer[self.nchannels - self.state] =
            self.buffer_process(self.nchannels - self.state);
        self.plan.process_outofplace_with_scratch(
            &mut self.pre_out_buffer,
            &mut self.out_buffer,
            &mut self.scratch_buffer,
        );
    }
}
