#include "../include/cinterface.cuh"
#include "../include/offline_channelizer.cuh"

extern "C"
{
    chann* chann_create(int nstreams, int nchann, int nsl, int ntap, complex<float> *coeff_arr)
    {
        return reinterpret_cast<chann*>(new channelizer(coeff_arr));
    }
    void chann_destroy(chann* inp)
    {
        delete reinterpret_cast<channelizer *>(inp);
    }

    void chann_process(chann* inp, complex<float>* lhs)
    {
        reinterpret_cast<channelizer*>(inp)->process(lhs);
    }
}