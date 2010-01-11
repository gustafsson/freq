#include "wavelettransform.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include "CudaException.h"
#include <math.h>
#include "CudaProperties.h"
#include "throwInvalidArgument.h"
#include <iostream>
#include "Statistics.h"
#include "StatisticsRandom.h"
#include <string.h>

// defined in wavlett.cu
void computeWavelettTransform( float* in_waveform_ft, float* out_waveform_ft, unsigned sampleRate, float minHz, float maxHz, cudaExtent numElem);
void inverseWavelettTransform( float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem);

WavelettTransform::WavelettTransform( const char* filename )
{
    _originalWaveform.reset( new Waveform( filename ));

    CudaProperties::printInfo(CudaProperties::getCudaDeviceProp());
}

boost::shared_ptr<TransformData> WavelettTransform::getWavelettTransform() {
    if (!_transform.get()) computeCompleteWavelettTransform();
    return _transform;
}

boost::shared_ptr<Waveform> WavelettTransform::getOriginalWaveform() {
    return _originalWaveform;
}

boost::shared_ptr<Waveform> WavelettTransform::getInverseWaveform() {
    if (!_inverseWaveform.get()) computeInverseWaveform();
    return _inverseWaveform;
}

void cufftSafeCall( cufftResult_t cufftResult) {
    if (cufftResult != CUFFT_SUCCESS) {
        ThrowInvalidArgument( cufftResult );
    }
}

boost::shared_ptr<TransformData> WavelettTransform::computeCompleteWavelettTransform()
{
    cudaExtent x = _originalWaveform->_waveformData->getNumberOfElements();
    x.width *= 2;
    x.height = 1;
    x.depth = 1;
    GpuCpuData<float> complexOriginal(0, x);
    float *original = _originalWaveform->_waveformData->getCpuMemory();
    float *complex = complexOriginal.getCpuMemory();
    for (int i=0; i<x.width/2; i++) {
        complex[i*2 + 0] = original[i];
        complex[i*2 + 1] = 0;
    }
    complexOriginal.getCudaGlobal();

TaskTimer tt(__FUNCTION__);

    TaskTimer* tt2 = new TaskTimer("start");

    _transform.reset( new TransformData());
    _transform->maxHz = _originalWaveform->_sample_rate/2;
    _transform->minHz = 20;
    _transform->sampleRate = _originalWaveform->_sample_rate;

    // Size of padded in-signal
    cudaExtent noe = _originalWaveform->_waveformData->getNumberOfElements();
    unsigned n = (1 << ((unsigned)ceil(log2(noe.width+1))));

    // Count number of scales to use
    float octaves = log2(_transform->maxHz)-log2(_transform->minHz);
    float granularity = 60; // scales per octave
    unsigned nFrequencies = granularity*octaves;

    // Allocate transform
    cudaExtent transformSize = {2*n, nFrequencies, 1 };
//    cudaExtent transformSize = {n, nFrequencies, _originalWaveform->channel_count() };
    _transform->transformData.reset( new GpuCpuData<float>( 0, transformSize, GpuCpuVoidData::CudaGlobal ));
    cudaMemset( _transform->transformData->getCudaGlobal().ptr(), 0, _transform->transformData->getSizeInBytes1D() );

    //    for (unsigned channel=0; channel<transformSize.depth; channel++) {
        // Padd in-signal
        GpuCpuData<float> waveform_ft((float*)0, make_cudaExtent(2*n,1,1), GpuCpuVoidData::CudaGlobal);
        cudaMemset( waveform_ft.getCudaGlobal().ptr(), 0, waveform_ft.getSizeInBytes1D());

        //cudaMemcpy( waveform_ft.getCudaGlobal().ptr(), _originalWaveform->_waveformData->getCpuMemory(), noe.width*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy( waveform_ft.getCudaGlobal().ptr()+2, complexOriginal.getCudaGlobal().ptr(), noe.width*sizeof(float)*2, cudaMemcpyDeviceToDevice);
        //int cW = memcmp(waveform_ft.getCpuMemory(), _originalWaveform->_waveformData->getCpuMemory(), noe.width*sizeof(float));

        // Transform signal
        cufftHandle plan;
        cufftSafeCall(cufftPlan1d(&plan, n, CUFFT_C2C, 1));

        //GpuCpuData<float> waveform_ft_wt(0, waveform_ft.getNumberOfElements(), GpuCpuVoidData::CudaGlobal);
  //      for (int j=0; j<10090; j++)
        // 1.28 s /10090 cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)waveform_ft.getCudaGlobal().ptr(), (cufftComplex *)waveform_ft_wt.getCudaGlobal().ptr(), CUFFT_FORWARD));
        cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)waveform_ft.getCudaGlobal().ptr(), (cufftComplex *)waveform_ft.getCudaGlobal().ptr(), CUFFT_FORWARD));
        //2.5 s / 10090 cufftSafeCall(cufftExecR2C(plan, (cufftReal *)waveform_ft.getCudaGlobal().ptr(), (cufftComplex *)waveform_ft_wt.getCudaGlobal().ptr()));


        //Destroy CUFFT context
        cufftDestroy(plan);
        CudaException_ThreadSynchronize();
delete tt2;
        // for each wanted scale
        //GpuCpuData<float> waveform_ft_wt(0, waveform_ft.getNumberOfElements(), GpuCpuVoidData::CudaGlobal);
        //cudaMemset( waveform_ft_wt.getCudaGlobal().ptr(), 0, waveform_ft_wt.getSizeInBytes1D());
        //cudaMemcpy( waveform_ft_wt.getCudaGlobal().ptr(), waveform_ft.getCudaGlobal().ptr(), waveform_ft_wt.getSizeInBytes1D(), cudaMemcpyDeviceToDevice);


        float* dp_data = _transform->transformData->getCudaGlobal().ptr();
        { TaskTimer tt("computing");
            ::computeWavelettTransform( waveform_ft.getCudaGlobal().ptr(),
                                        dp_data,
                                        _transform->sampleRate,
                                        _transform->minHz,
                                        _transform->maxHz,
                                        _transform->transformData->getNumberOfElements()
                                        );
            CudaException_ThreadSynchronize(); }

        {        TaskTimer tt("inverse fft");

            // Transform signal back
            cufftSafeCall(cufftPlan1d(&plan, n, CUFFT_C2C, nFrequencies));
            cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)dp_data, (cufftComplex *)dp_data, CUFFT_INVERSE));

            //Destroy CUFFT context
            cufftDestroy(plan);
            CudaException_ThreadSynchronize(); }
//    } // channel

    CudaException_ThreadSynchronize();

    return _transform;
}

boost::shared_ptr<Waveform> WavelettTransform::computeInverseWaveform()
{
    {
    TaskTimer tt(__FUNCTION__);
    //cudaExtent sz = _originalWaveform->_waveformData->getNumberOfElements();
    cudaExtent sz = getWavelettTransform()->transformData->getNumberOfElements();

    sz.height = 1;
    sz.depth = 1;
    sz.width = _originalWaveform->_waveformData->getNumberOfElements().width;

    // summarize them all
    _inverseWaveform.reset( new Waveform() );
    _inverseWaveform->_sample_rate = _originalWaveform->_sample_rate;
    _inverseWaveform->_waveformData.reset( new GpuCpuData<float>(0, sz, GpuCpuVoidData::CudaGlobal) );

    ::inverseWavelettTransform(
            getWavelettTransform()->transformData->getCudaGlobal().ptr(),
            getWavelettTransform()->transformData->getNumberOfElements(),
            _inverseWaveform->_waveformData->getCudaGlobal().ptr(),
            _inverseWaveform->_waveformData->getNumberOfElements()
    );
}
        {
    TaskTimer tt("inverse corollary");

    size_t n = _inverseWaveform->_waveformData->getNumberOfElements1D();
    float* data = _inverseWaveform->_waveformData->getCpuMemory();
    float* orgdata = _originalWaveform->_waveformData->getCpuMemory();
    double sum = 0, orgsum=0;
    for (size_t i=0; i<n; i++) {
        sum += fabsf(data[i]);
    }
    for (size_t i=0; i<n; i++) {
        orgsum += fabsf(orgdata[i]);
    }
    float scale = orgsum/sum;
    for (size_t i=0; i<n; i++)
        data[i]*=scale;
    tt.info("scales %g, %g, %g", sum, orgsum, scale);

    _inverseWaveform->writeFile("outtest.wav");
}
    return _inverseWaveform;
}

boost::shared_ptr<TransformData> WavelettTransform::computeWavelettTransform( float startt, float endt, float lowf, float highf, unsigned numf )
{
    return boost::shared_ptr<TransformData>();
}
