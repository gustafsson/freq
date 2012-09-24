#include "complexbuffer.h"

#include "TaskTimer.h"
#include "cpumemorystorage.h"

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
        ComplexBuffer(const Signal::MonoBuffer& buffer)
            :
            sample_offset(buffer.sample_offset()),
            sample_rate(buffer.sample_rate())
{
    setData(buffer.waveform_data());
}


ComplexBuffer::
        ComplexBuffer(DataStorage<float>::Ptr real_waveform)
            :
            sample_offset(0),
            sample_rate(1)
{
    setData(real_waveform);
}


void ComplexBuffer::
        setData(DataStorage<float>::Ptr real_waveform)
{
    DataStorageSize sz = real_waveform->size();
    TIME_COMPLEX_BUFFER TaskTimer tt("ComplexBuffer of %lu x %lu x %lu elements", sz.width, sz.height, sz.depth );

    _complex_waveform_data.reset( new DataStorage<std::complex<float> >( sz ));

    std::complex<float>*complex = _complex_waveform_data->getCpuMemory();
    float *real = real_waveform->getCpuMemory();

    for (int z=0; z<sz.depth; z++)
        for (int y=0; y<sz.height; y++)
        {
            int o = (y + z*sz.height)*sz.width;

            for (int x=0; x<sz.width; x++)
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


Signal::pMonoBuffer ComplexBuffer::
        get_real()
{
    Signal::IntervalType length = number_of_samples();
    Signal::pMonoBuffer buffer( new Signal::MonoBuffer( sample_offset, length, sample_rate ));

    DataStorage<float>::Ptr real_waveform = buffer->waveform_data();

    DataStorageSize sz = real_waveform->size();
    std::complex<float> *complex = CpuMemoryStorage::ReadOnly<1>( _complex_waveform_data ).ptr();
    float *real = CpuMemoryStorage::WriteAll<1>( real_waveform ).ptr();

    for (int z=0; z<sz.depth; z++)
        for (int y=0; y<sz.height; y++)
            for (int x=0; x<sz.width; x++)
            {
                Signal::IntervalType o = x + (y + z*sz.height)*sz.width;
                real[ o ] = complex[ o ].real(); // discard .y component
            }

    return buffer;
}

} // namespace Tfr
