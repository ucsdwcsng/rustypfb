#ifndef _CHANNELIZER_
#define _CHANNELIZER_
#include <cufft.h>
#include <complex.h>
#include <memory>
#include <vector>

using std::complex;
using std::unique_ptr;
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::weak_ptr;
using std::make_unique;
using std::make_shared;

void __global__ create_polyphase_input(cufftComplex*, cufftComplex*, int, int);
void __global__ multiply(cufftComplex*, cufftComplex*, cufftComplex*, int);
void __global__ make_coeff_matrix(complex<float>*, cufftComplex*, int, int, int);

/*
 * This struct wraps together CUFFT plans and the CUDA streams that those plans execute on.
 * Useful to launch CUFFT methods asynchronously on different streams.
 */

struct ProcessData {
    cufftHandle plan;
    int istride;
    int ostride;
    int idist;
    int odist;
    int batch;
    int* inembed;
    int* onembed;
    int* n;
    int rank;
    weak_ptr<cudaStream_t> stream;
    ProcessData(int, int, int, int, int, int, int,shared_ptr<cudaStream_t>);
    ~ProcessData();
};

class channelizer
{
    public:
    /*
     * Polyphase filter coefficients which have been filtered along slice dimension.
     */
    int rank = 1;

    /*
     * Plan for taking FFT along slice dimension. This requires no reshape of input.
     */

    cufftHandle plan_1;
    int istride_1;
    int ostride_1;
    int idist_1;
    int odist_1;
    int batch_1;
    int* inembed_1;
    int* onembed_1;
    int* n_1;

    // /*
    //  * Plan for taking IFFT along channels.
    //  */

    cufftHandle plan;
    int istride;
    int ostride;
    int idist;
    int odist;
    int batch;
    int* inembed;
    int* onembed;
    int* n;

    /*
     * Plan for FFT along channel dimensions
     */
    cufftHandle plan_0;
    int istride_0;
    int ostride_0;
    int idist_0;
    int odist_0;
    int batch_0;
    int* inembed_0;
    int* onembed_0;
    int* n_0;

    /*
     * Holds the Coefficient filter coefficients.
     */
    cufftComplex* coeff_fft_polyphaseform;

    /*
     * Output buffer to hold results.
     */
    cufftComplex* output_buffer;

    /*
     * Internal scratch buffer on GPU
     */
    cufftComplex* scratch_buffer;

    /*
     * Page locked memory on host which is accessible to both 
     * device and host.
     */
    cufftComplex* locked_buffer;

    // vector<shared_ptr<cudaStream_t>> streams;
    // vector<shared_ptr<ProcessData>> forward_process_fft_streams;

    /*
     * Constructor
     */
    channelizer(complex<float>*);
    void process(complex<float>*);
    ~channelizer();
};

unique_ptr<channelizer> create_chann(int, vector<complex<float>>);

#endif

