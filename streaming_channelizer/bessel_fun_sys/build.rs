use cc;
fn main() {
    // println!("cargo:rustc-link-lib=cudart");
    // println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("../bessel/src/c_interface.cpp")
        .compile("foo");
    // println!("cargo:rustc-link-lib=dylib=cudart");
}