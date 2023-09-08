#include "../include/offline_channelizer.cuh"
#include <iostream>
using std::cout;
using std::endl;
using std::make_unique;

void __global__ create_polyphase_input(cufftComplex *inp, cufftComplex *outp, int nchannel, int nslice)
{
    int slice_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (slice_id < nslice)
    {
        for (int channel_id = 0; channel_id < nchannel; channel_id++)
        {
            outp[channel_id * nslice + slice_id] = inp[(1 + slice_id) * nchannel - 1 - channel_id];
        }
    }
}

void __global__ dynamic_polyphase_input(cufftComplex *inp, cufftComplex *outp, int nchannel, int nslice)
{
    int channel_id = blockIdx.x * blockDim.x + threadIdx.x;
    int slice_id = blockIdx.y * blockDim.y + threadIdx.y;

    if ((slice_id < nslice) && (channel_id < nchannel))
    {
        outp[channel_id * nslice + slice_id] = inp[(1 + slice_id) * nchannel - 1 - channel_id];
    }
}

void __global__ multiply(cufftComplex *inp, cufftComplex *coeff, cufftComplex *outp, int nsamples)
{
    int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id < nsamples)
        outp[sample_id] = make_cuComplex(inp[sample_id].x * coeff[sample_id].x - inp[sample_id].y * coeff[sample_id].y, inp[sample_id].x * coeff[sample_id].y + inp[sample_id].y * coeff[sample_id].x);
}

void make_coeff_matrix(cufftComplex* gpu, complex<float>* inp, int nchannel, int ntaps, int nslice) {
    for (int id = 0; id < nchannel * ntaps; id++)
    {
        int tap_id = id / nchannel;
        int chann_id = id % nchannel;
        cudaMemcpy(gpu + tap_id * nchannel + chann_id * ntaps, inp + id, sizeof(cufftComplex), cudaMemcpyHostToDevice);
        auto err_0 = cudaGetLastError();
        // cout << "Memcpy2d error " << cudaGetErrorString(err_0) << endl;
    }
    // int copy_width = nchannel * sizeof(complex<float>);
    // int copy_height = ntaps;
    // int spitch = nchannel * sizeof(complex<float>);
    // int dpitch = nslice * sizeof(cufftComplex);
    // cudaMemcpy2D(gpu, dpitch, inp, spitch, copy_width, copy_height, cudaMemcpyHostToDevice);
}

void make_reverse_coeff_matrix(cufftComplex* gpu, complex<float>* inp, int nchannel, int ntaps) {
    for (int id = 0; id < nchannel * ntaps; id++)
    {
        int tap_id = id / nchannel;
        int chann_id = id % nchannel;
        cudaMemcpy(gpu + tap_id * nchannel + (nchannel - chann_id) * ntaps, inp + id, sizeof(cufftComplex), cudaMemcpyHostToDevice);
        auto err_0 = cudaGetLastError();
        cout << "Memcpy2d error " << cudaGetErrorString(err_0) << endl;
    }
    // int copy_width = nchannel * sizeof(complex<float>);
    // int copy_height = ntaps;
    // int spitch = nchannel * sizeof(complex<float>);
    // int dpitch = nslice * sizeof(cufftComplex);
    // cudaMemcpy2D(gpu, dpitch, inp, spitch, copy_width, copy_height, cudaMemcpyHostToDevice);
}

void __global__ dynamic_multiply(cufftComplex *inp, cufftComplex *coeff, cufftComplex *outp, int nchannels, int nslices)
{
    int channel_id = blockIdx.x * blockDim.x + threadIdx.x;
    int slice_id = blockIdx.y * blockDim.y + threadIdx.y;

    if ((channel_id < nchannels) && (slice_id < nslices))
    {   
        auto sample_id = channel_id*nslices + slice_id;
        cufftComplex lhs  = inp[sample_id];
        cufftComplex rhs  = coeff[sample_id];
        outp[sample_id] = make_cuComplex(lhs.x* rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x);
    }
}
channelizer::channelizer(int nchann, int nsl, int ntap, complex<float> *coeff_arr)
{
    nchannel = nchann;
    nslice = nsl;
    ntaps = ntap;

    // Allocate GPU memory for filter coefficients.
    cudaMalloc((void **)&coeff_fft_polyphaseform, sizeof(cufftComplex) * nchannel * nslice);

    // Allocate GPU memory for input buffer.
    cudaMalloc((void **)&input_buffer, sizeof(cufftComplex) * nchannel * nslice);

    // Allocate GPU memory for internal buffer.
    cudaMalloc((void **)&internal_buffer, sizeof(cufftComplex) * nchannel * nslice);

    /*
     * Plan 1 : Take FFT along each row. There are nslice elements in each row.
     * There are nchannel rows.
     */
    istride_1 = 1;
    idist_1 = nslice;
    batch_1 = nchannel;
    ostride_1 = 1;
    odist_1 = nslice;
    n_1 = new int[1];
    *n_1 = nslice;
    inembed_1 = n_1;
    onembed_1 = n_1;

    /*
     * Plan 2 : Take IFFT along each row. There are nslice elements in each row.
     * There are nchannel rows.
     */
    // istride_2 = 1;
    // idist_2 = nslice;
    // batch_2 = nchannel;
    // ostride_2 = 1;
    // odist_2 = nslice;

    /*
     * Plan 3 : Take IFFT along each column. There are nslice elements in each row.
     * There are nchannel rows.
     */
    istride_2 = nslice;
    idist_2 = 1;
    batch_2 = nslice;
    ostride_2 = nslice;
    odist_2 = 1;
    n_2 = new int[1];
    *n_2 = nchannel;
    inembed_2 = n_2;
    onembed_2 = n_2;

    /*
     * Plan 4 : Take FFT along each coloumn. This will be used to reduce GPU memory accesses. No reshaing of input required.
     * There are nslice elements in each column.
     */
    istride_4 = nchannel;
    idist_4 = 1;
    batch_4 = nchannel;
    ostride_4 = nchannel;
    odist_4 = 1;
    n_4 = new int[1];
    *n_4 = nslice;
    inembed_4 = n_4;
    onembed_4 = n_4;

    istride_5 = 1;
    ostride_5 = 1;
    idist_5 = nchannel;
    odist_5 = nchannel;
    batch_5 = nslice;
    n_5 = new int [1];
    *n_5 = nchannel;
    inembed_5 = n_5;
    onembed_5 = n_5;
    

    cufftPlanMany(&plan_1, rank, n_1, inembed_1, istride_1, idist_1, onembed_1, ostride_1, odist_1, CUFFT_C2C, batch_1);
    cufftPlanMany(&plan_2, rank, n_2, inembed_2, istride_2, idist_2, onembed_2, ostride_2, odist_2, CUFFT_C2C, batch_2);
    cufftPlanMany(&plan_4, rank, n_4, inembed_4, istride_4, idist_4, onembed_4, ostride_4, odist_4, CUFFT_C2C, batch_4);
    cufftPlanMany(&plan_5, rank, n_5, inembed_5, istride_5, idist_5, onembed_5, ostride_5, odist_5, CUFFT_C2C, batch_5);


    // make_coeff_matrix(coeff_fft_polyphaseform, coeff_arr, nchannel, ntaps, nslice);
    make_reverse_coeff_matrix(coeff_fft_polyphaseform, coeff_arr, nchannel, ntaps);

    // Store the Slice FFT of the coefficient matrix in the start itself.
    // cufftExecC2C(plan_1, coeff_fft_polyphaseform, coeff_fft_polyphaseform, CUFFT_FORWARD);

    // Hopefully faster GPU version.
    cufftExecC2C(plan_4, coeff_fft_polyphaseform, coeff_fft_polyphaseform, CUFFT_FORWARD);
}

void channelizer::process(complex<float>* input, complex<float>* output)
{
    cudaMemcpy(input_buffer, input, sizeof(cufftComplex)*nchannel*nslice, cudaMemcpyHostToDevice);
    auto err_1 = cudaGetLastError();
    // cout << "Memcpy error" << cudaGetErrorString(err_1) << endl;

    // create_polyphase_input<<<nchannel, nslice>>>(input_buffer, internal_buffer, nchannel, nslice);
    // auto err_2 = cudaGetLastError();
    // cout << "Polyphase error" << cudaGetErrorString(err_2) << endl;

    dim3 dimBlock(16, 32);
    dim3 dimGrid(1024, nslice / 16);
    // dynamic_polyphase_input<<<dimGrid, dimBlock>>>(input_buffer, internal_buffer, nchannel, nslice);
    // auto err_2 = cudaGetLastError();
    // cout << "Polyphase error" << cudaGetErrorString(err_2) << endl;

    // /*
    //  * FFT along slice dimension of the polyphas inputs.
    //  */
    // cufftExecC2C(plan_1, internal_buffer, internal_buffer, CUFFT_FORWARD);
    

    // Execute FFT along each coloumn.
    cufftExecC2C(plan_4, input_buffer, input_buffer, CUFFT_FORWARD);
    auto err_3 = cudaGetLastError();
    // cout << "Fft error" << cudaGetErrorString(err_3) << endl;

    /*
     * Multiply the FFT of input and FFT of filter coefficients.
     */
    // for (int chann_id = 0; chann_id < nchannel; chann_id++)
    // {
    //     multiply<<<nslice, 2>>>(input_buffer + chann_id * nslice, coeff_fft_polyphaseform + chann_id * nslice, internal_buffer + chann_id * nslice, nslice);
    //     auto err_4 = cudaGetLastError();
    //     cout << "Multiply Error" << cudaGetErrorString(err_4) << endl;
    // }
    // dynamic_multiply<<<dimGrid, dimBlock>>>(internal_buffer, coeff_fft_polyphaseform, internal_buffer, nchannel, nslice);
    // auto err_ = cudaGetLastError();
    // cout << "Multiply_error" << cudaGetErrorString(err_) << endl;
    dynamic_multiply<<<dimGrid, dimBlock>>>(input_buffer, coeff_fft_polyphaseform, internal_buffer, nchannel, nslice);
    auto err_ = cudaGetLastError();
    // cout << "Multiply_error" << cudaGetErrorString(err_) << endl;

    // /*
    //  * This is the IFFT of the product and represents the convolution of
    //  * each polyphase component of the filter with the input.
    //  */
    // cufftExecC2C(plan_1, internal_buffer, internal_buffer, CUFFT_INVERSE);
    // auto err_4 = cudaGetLastError();
    // cout << "Fft 2 error" << cudaGetErrorString(err_4) << endl;

    cufftExecC2C(plan_4, internal_buffer, internal_buffer, CUFFT_INVERSE);
    auto err_4 = cudaGetLastError();
    // cout << "Fft 2 error" << cudaGetErrorString(err_4) << endl;

    /*
     * Final IFFT representing the downconversion 
     */
    cufftExecC2C(plan_5, internal_buffer, internal_buffer, CUFFT_INVERSE);
    auto err_5 = cudaGetLastError();
    // cout << "Fft 3 error" << cudaGetErrorString(err_5) << endl;

    /*
     * Send to output buffer.
     */
    cudaMemcpy(output, internal_buffer, sizeof(complex<float>)*nchannel*nslice, cudaMemcpyDeviceToHost);
    auto err_6 = cudaGetLastError();
    // cout << "Memcpy error" << cudaGetErrorString(err_6) << endl;
}

channelizer::~channelizer()
{
    cufftDestroy(plan_1);
    cufftDestroy(plan_2);
    cufftDestroy(plan_4);
    cufftDestroy(plan_5);
    delete [] n_1;
    delete [] n_2;
    delete [] n_4;
    delete [] n_5;

    cudaFree(coeff_fft_polyphaseform);
    cudaFree(internal_buffer);
    cudaFree(input_buffer);
}

unique_ptr<channelizer> create_chann(int nchann, int nsl, int ntap, vector<complex<float>> coeff_arr)
{
    return make_unique<channelizer>(nchann, nsl, ntap, &coeff_arr[0]);
}