use libm::sinf;
use libm::sqrtf;
use num::complex::Complex64;
use num::Complex;
use bessel_fun_sys::bessel_func;
use rayon::prelude::*;
use rustfft::algorithm::Radix4;
use rustfft::Fft;
use std::collections::VecDeque;
use std::f32::consts::PI;

pub struct Queue<T> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T> Queue<T> {
    pub fn new(capacity: usize) -> Self {
        Queue {
            buffer: VecDeque::new(),
            capacity: capacity,
        }
    }

    pub fn add(&mut self, element: T) {
        if self.buffer.len() < self.capacity {
            self.buffer.push_back(element);
        } else {
            self.buffer.pop_front();
            self.buffer.push_back(element);
        }
    }
}

/*
 Streaming version of the polyphase channelizer.
 Updates outputs one slice across channels at a time.
 This is maximally decimating.
*/
pub struct StreamChannelizer {
    // This is the number of channels.
    nchannels: usize,

    // This is the number of taps per channel.
    ntaps: usize,

    // Filter coefficients in polyphase form
    coeff: Vec<Vec<Complex<f32>>>,

    // Plan for IFFT
    plan: Radix4<f32>,

    // Collection of FIFO queues for each channel. Each buffer has capacity = taps.
    internal_buffers: Vec<Queue<Complex<f32>>>,

    // pre_out_buffer
    pre_out_buffer: Vec<Complex<f32>>,

    // // Output Buffer : This represents the current one slice output across channels
    // out_buffer: Vec<Complex<f32>>,

    // Scratch buffer for FFT
    scratch_buffer: Vec<Complex<f32>>,
}

pub fn create_filter(taps: usize, channels: usize) -> Vec<Vec<Complex<f32>>> {
    let mut inter_buffer: Vec<Vec<Complex<f32>>> = Vec::with_capacity(channels);
    let channel_half = channels / 2;
    for chann_id in 0..channels {
    let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); 2 * taps];
        for tap_id in 0..taps {
        let ind = tap_id*channels + chann_id;
        if chann_id < channel_half {
            buffer[2 * tap_id] = Complex::new(channel_fn(ind, channels, taps, 10.0), 0.0);
        } else {
            buffer[2 * tap_id + 1] = Complex::new(channel_fn(ind, channels, taps, 10.0), 0.0);
        }
    }
    inter_buffer.push(buffer);
    }
    inter_buffer
}

pub fn create_state(taps: usize, channels: usize) -> Vec<Queue<Complex<f32>>> {
    let mut outp: Vec<Queue<Complex<f32>>> = Vec::new();
    for chann in 0..(channels / 2) {
        outp.push(Queue::new(2 * taps));
    }
    outp
}

pub fn channel_fn(ind: usize, nchannel: usize, nproto: usize, kbeta: f32) -> f32 {
    let ind_arg = ind as f32;
    let arg = -((nproto / 2) as f32) + (ind_arg + 1.0) / (nchannel as f32);
    let darg = (2.0 * ind_arg) / ((nchannel * nproto) as f32) - 1.0;
    let carg = kbeta * sqrtf(1.0 - darg * darg);
    (unsafe { bessel_func(carg) }) / (unsafe { bessel_func(kbeta) })
        * (if arg != 0.0 { sinf(arg) / arg } else { 1.0 })
}

impl StreamChannelizer {
    pub fn new(taps: usize, channels: usize) -> Self {
        StreamChannelizer {
            nchannels: channels,
            ntaps: taps,
            coeff: create_filter(taps, channels),
            plan: Radix4::new(channels, rustfft::FftDirection::Inverse),
            internal_buffers: create_state(taps, channels),
            pre_out_buffer: vec![Complex::new(0.0, 0.0); channels],
            scratch_buffer: vec![Complex::new(0.0, 0.0); channels],
        }
    }
    pub fn process(&mut self, sample_arr: &[Complex<f32>], output_buffer: &mut [Complex<f32>]) {
        self.internal_buffers
            .par_iter_mut()
            .enumerate()
            .for_each(|(ind, item)| item.add(sample_arr[self.nchannels / 2 - ind - 1]));

        (self.pre_out_buffer)
            .par_iter_mut()
            .enumerate()
            .for_each(|(ind, item)| {
                (*item) = buffer_process(&self.internal_buffers, &self.coeff, ind)
            });
        self.plan.process_outofplace_with_scratch(
            &mut self.pre_out_buffer,
            output_buffer,
            &mut self.scratch_buffer,
        )
    }
}

pub fn buffer_process(
    lhs: &Vec<Queue<Complex<f32>>>,
    rhs: &Vec<Vec<Complex<f32>>>,
    id: usize,
) -> Complex<f32> {
    let mut sum = Complex { re: 0.0, im: 0.0 };
    let nchannels = rhs.len();
    println!("{}", nchannels);
    let reduced_ind = if id < nchannels / 2 {
        id
    } else {
        id - nchannels / 2
    };
    for (ind, item) in lhs[reduced_ind].buffer.iter().enumerate(){
        sum += (*item) * rhs[id][ind];
    }
    sum
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let channels:usize = 1024;
        let taps : usize= 128;

        let mut channelizer = StreamChannelizer::new(128, 1024);

        // Example input signal: Let's just use a bunch of 1's for simplicity
        let input_signal = vec![Complex::new(1.0 as f32, 0.0); channels / 2];

        // Buffer for the channelizer output
        let mut output_buffer = vec![Complex::new(0.0 as f32, 0.0); channels];

        // Process the input signal
        channelizer.process(&input_signal, &mut output_buffer);

        // Print the output buffer
        for (i, sample) in output_buffer.iter().enumerate() {
            println!("Channel {}: {:?}", i, sample);
        }
    }
}
