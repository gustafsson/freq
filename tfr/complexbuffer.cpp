#include "complexbuffer.h"

namespace Tfr
{

ComplexBuffer::ComplexBuffer(unsigned first_sample, unsigned numberOfSamples, unsigned FS, unsigned numberOfChannels)
:   sample_offset(first_sample),
    sample_rate(FS)
{
    if (numberOfSamples)
        _complex_waveform_data.reset( new GpuCpuData<float2>(0, make_cudaExtent( numberOfSamples, numberOfChannels, 1)));
}

ComplexBuffer::
        ComplexBuffer(const Signal::Buffer& buffer)
            :
            sample_offset(buffer.sample_offset),
            sample_rate(buffer.sample_rate)
{
    GpuCpuData<float>* real_waveform = buffer.waveform_data();
    cudaExtent sz = real_waveform->getNumberOfElements();

    _complex_waveform_data.reset( new GpuCpuData<float2>( 0, sz ));

    float2 *complex = _complex_waveform_data->getCpuMemory();
    float *real = real_waveform->getCpuMemory();

    for (unsigned z=0; z<sz.depth; z++)
        for (unsigned y=0; y<sz.height; y++)
            for (unsigned x=0; x<sz.width; x++)
            {
                unsigned o = x + (y + z*sz.height)*sz.width;
                // set .y component to 0
                complex[ o ] = make_float2( real[ o ], 0 );
            }
}


GpuCpuData<float>* ComplexBuffer::
        waveform_data()
{
    _my_real.reset();
    _my_real = get_real();
    return _my_real->waveform_data();
}


Signal::pBuffer ComplexBuffer::
        get_real()
{
    unsigned length = number_of_samples();
    Signal::pBuffer buffer( new Signal::Buffer( sample_offset, length, sample_rate ));

    GpuCpuData<float>* real_waveform = buffer->waveform_data();

    cudaExtent sz = real_waveform->getNumberOfElements();
    float2 *complex = _complex_waveform_data->getCpuMemory();
    float *real = real_waveform->getCpuMemory();

    for (unsigned z=0; z<sz.depth; z++)
        for (unsigned y=0; y<sz.height; y++)
            for (unsigned x=0; x<sz.width; x++)
            {
                unsigned o = x + (y + z*sz.height)*sz.width;
                real[ o ] = complex[ o ].x; // discard .y component
            }

    return buffer;
}

} // namespace Tfr
