#include "../include/offline_chann_C_interface.cuh"

extern "C"
{
    chann* chann_create(complex<float>* coeff_arr)
    {
        return reinterpret_cast<chann*>(new channelizer(coeff_arr));
    }

    void chann_destroy(chann* inp)
    {
        delete reinterpret_cast<channelizer*>(inp);
    }

    void chann_process(chann* chann, float* lhs, complex<float>* rhs)
    {
        reinterpret_cast<channelizer*>(chann)->process(lhs, rhs);
    }

    complex<float>* memory_allocate(int size)
    {
        complex<float>* output;
        cudaMalloc((void**)&output, sizeof(complex<float>)*size);
        return output;
    }
}