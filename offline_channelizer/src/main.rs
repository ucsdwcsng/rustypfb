use cxx::UniquePtr;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++"{
        include!("../include/offline_channelizer.cuh");
        type channelizer;
        fn create_chann(nchann: i32, nchannels: i32, ntaps:i32, ) -> UniquePtr<channelizer>;
    }
}

fn main() {
    println!("Hello World!");
}