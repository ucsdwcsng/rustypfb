use libm::cos;
use libm::erfc;
use libm::log;
use libm::sin;
use libm::sqrt;
use num::complex::Complex64;
use num::Complex;
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
pub struct Channelizer {
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
    for ind in 0..channels * taps {
        let tap_id = ind / channels;
        let chann_id = ind % channels;
        let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); 2 * taps];
        if chann_id < channel_half {
            buffer[2 * tap_id] = Complex::new(channel_fn(ind), 0.0);
        } else {
            buffer[2 * tap_id + 1] = Complex::new(channel_fn(ind), 0.0);
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

pub fn channel_fn(ind: usize) -> f32 {
    todo!();
}

impl Channelizer {
    pub fn new(taps: usize, channels: usize) -> Self {
        Channelizer {
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
            .for_each(|(ind, item)| item.add(sample_arr[self.nchannels / 2 - ind]));

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
    let reduced_ind = if id < nchannels / 2 {
        id
    } else {
        id - nchannels / 2
    };
    for (ind, item) in rhs[id].iter().enumerate() {
        sum += (*item) * (lhs[reduced_ind].buffer)[ind];
    }
    sum
}
