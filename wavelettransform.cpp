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
void computeWavelettTransform( float* in_waveform_ft, float* out_waveform_ft, float period, unsigned numElem);
void inverseWavelettTransform( float* in_wavelett_ft, float* out_inverse_waveform, cudaExtent numElem);

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
    // Assume signal and filter kernel are padded

    _transform.reset( new TransformData());
    _transform->maxHz = _originalWaveform->_sample_rate/2;
    _transform->minHz = 20;
    _transform->sampleRate = _originalWaveform->_sample_rate;

    // Size of padded in-signal
    cudaExtent noe = _originalWaveform->_waveformData->getNumberOfElements();
    unsigned n = 2 * (1 << ((unsigned)ceil(log2(noe.width))));

    cufftHandle plan;
    cufftSafeCall(cufftPlan1d(&plan, n/2, CUFFT_C2C, 1));

    // Count number of scales to use
    float octaves = log2(_transform->maxHz)-log2(_transform->minHz);
    float granularity = 100; // scales per octave
    unsigned nFrequencies = granularity*octaves;

    // Allocate transform
    cudaExtent transformSize = {n, nFrequencies, 1 };
//    cudaExtent transformSize = {n, nFrequencies, _originalWaveform->channel_count() };
    _transform->transformData.reset( new GpuCpuData<float>( 0, transformSize, GpuCpuVoidData::CudaGlobal ));
    cudaMemset( _transform->transformData->getCudaGlobal().ptr(), 0, _transform->transformData->getSizeInBytes1D() );

    for (unsigned channel=0; channel<transformSize.depth; channel++)
    {
        // Padd in-signal
        GpuCpuData<float> waveform_ft((float*)0, make_cudaExtent(n,1,1), GpuCpuVoidData::CudaGlobal);
        cudaMemset( waveform_ft.getCudaGlobal().ptr(), 0, waveform_ft.getSizeInBytes1D());

        cudaMemcpy( waveform_ft.getCudaGlobal().ptr(), _originalWaveform->_waveformData->getCpuMemory(), noe.width*sizeof(float), cudaMemcpyHostToDevice);
        int cW = memcmp(waveform_ft.getCpuMemory(), _originalWaveform->_waveformData->getCpuMemory(), noe.width*sizeof(float));

        // Transform signal
        cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)waveform_ft.getCudaGlobal().ptr(), (cufftComplex *)waveform_ft.getCudaGlobal().ptr(), CUFFT_FORWARD));
        //cufftSafeCall(cufftExecR2C(plan, (cufftReal *)waveform_ft.getCudaGlobal().ptr(), (cufftComplex *)waveform_ft.getCudaGlobal().ptr()));

        CudaException_ThreadSynchronize("NOOO2");

        // for each wanted scale
        GpuCpuData<float> waveform_ft_wt(0, waveform_ft.getNumberOfElements(), GpuCpuVoidData::CudaGlobal);
        cudaMemset( waveform_ft_wt.getCudaGlobal().ptr(), 0, waveform_ft_wt.getSizeInBytes1D());
        //cudaMemcpy( waveform_ft_wt.getCudaGlobal().ptr(), waveform_ft.getCudaGlobal().ptr(), waveform_ft_wt.getSizeInBytes1D(), cudaMemcpyDeviceToDevice);

        for (unsigned fi = 0; fi<nFrequencies; fi++)
        {

            float f = _transform->getFrequency(fi);
            float period = _transform->sampleRate/f;

            computeWavelettTransform( waveform_ft.getCudaGlobal().ptr(), waveform_ft_wt.getCudaGlobal().ptr(), period/n, n);

            CudaException_ThreadSynchronize("NOOO");
            // Transform signal back
            unsigned offset = fi*n + channel*n*nFrequencies;
            cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)waveform_ft_wt.getCudaGlobal().ptr(), (cufftComplex *)(_transform->transformData->getCudaGlobal().ptr()+offset), CUFFT_INVERSE));
       }
    }

    //Destroy CUFFT context
    cufftDestroy(plan);

    CudaException_CHECK_ERROR();
           CudaException_ThreadSynchronize("JAA");
}

boost::shared_ptr<Waveform> WavelettTransform::computeInverseWaveform()
{
    //cudaExtent sz = _originalWaveform->_waveformData->getNumberOfElements();
    cudaExtent sz = getWavelettTransform()->transformData->getNumberOfElements();

    sz.height = 1;
    sz.depth = 1;

    // summarize them all
    _inverseWaveform.reset( new Waveform() );
    _inverseWaveform->_sample_rate = _originalWaveform->_sample_rate;
    _inverseWaveform->_waveformData.reset( new GpuCpuData<float>(0, sz, GpuCpuVoidData::CudaGlobal) );

    size_t t = getWavelettTransform()->transformData->getSizeInBytes1D();
    size_t w = _inverseWaveform->_waveformData->getSizeInBytes1D();
    cudaExtent x = getWavelettTransform()->transformData->getNumberOfElements();
/*    inverseWavelettTransform(
            getWavelettTransform()->transformData->getCudaGlobal().ptr(),
            _inverseWaveform->_waveformData->getCudaGlobal().ptr(),
            getWavelettTransform()->transformData->getNumberOfElements()
            );
*/
}
