#include "complexbuffer.h"

#include "TaskTimer.h"

#define TIME_COMPLEX_BUFFER if(0)
//#define TIME_COMPLEX_BUFFER

namespace Tfr
{

ComplexBuffer::ComplexBuffer(UnsignedF first_sample, unsigned long numberOfSamples, float FS, unsigned numberOfChannels)
:   sample_offset(first_sample),
    sample_rate(FS)
{
    if (numberOfSamples)
        _complex_waveform_data.reset( new DataStorage<std::complex<float> >( DataStorageSize (numberOfSamples, numberOfChannels, 1)));
}


ComplexBuffer::
        ComplexBuffer(const Signal::Buffer& buffer)
            :
            sample_offset(buffer.sample_offset),
            sample_rate(buffer.sample_rate)
{
    DataStorage<float>::Ptr real_waveform = buffer.waveform_data();
    DataStorageSize sz = real_waveform->getNumberOfElements();
    TIME_COMPLEX_BUFFER TaskTimer tt("ComplexBuffer of %lu x %lu x %lu elements", sz.width, sz.height, sz.depth );

    _complex_waveform_data.reset( new DataStorage<std::complex<float> >( sz ));

    std::complex<float>*complex = _complex_waveform_data->getCpuMemory();
    float *real = real_waveform->getCpuMemory();

    for (unsigned z=0; z<sz.depth; z++)
        for (unsigned y=0; y<sz.height; y++)
        {
            unsigned o = (y + z*sz.height)*sz.width;

            for (unsigned x=0; x<sz.width; x++)
            {
                // set .y component to 0
                complex[ x + o ] = std::complex<float>( real[ x + o ], 0 );
            }
        }
}


DataStorage<float>::Ptr ComplexBuffer::
        waveform_data()
{
    _my_real.reset();
    _my_real = get_real();
    return _my_real->waveform_data();
}


Signal::pBuffer ComplexBuffer::
        get_real()
{
    Signal::IntervalType length = number_of_samples();
    Signal::pBuffer buffer( new Signal::Buffer( sample_offset, length, sample_rate ));

    DataStorage<float>::Ptr real_waveform = buffer->waveform_data();

    DataStorageSize sz = real_waveform->getNumberOfElements();
    std::complex<float> *complex = _complex_waveform_data->getCpuMemory();
    float *real = real_waveform->getCpuMemory();

    for (unsigned z=0; z<sz.depth; z++)
        for (unsigned y=0; y<sz.height; y++)
            for (size_t x=0; x<sz.width; x++)
            {
                Signal::IntervalType o = x + (y + z*sz.height)*sz.width;
                real[ o ] = complex[ o ].real(); // discard .y component
            }

    return buffer;
}

} // namespace Tfr
