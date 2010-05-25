#include "signal-source.h"
#include <string.h>
#include <stdio.h>
#include "signal-samplesintervaldescriptor.h"

using namespace std;

namespace Signal {

Buffer::Buffer(Interleaved interleaved)
:   sample_offset(0),
    sample_rate(0),
    _interleaved(interleaved)
{
    switch(_interleaved) {
        case Interleaved_Complex:
        case Only_Real:
            break;
        default:
            throw invalid_argument( string( __FUNCTION__ ));
    }
}

Buffer::Buffer(unsigned first_sample, unsigned numberOfSamples, unsigned FS, Interleaved interleaved)
:   sample_offset(first_sample),
    sample_rate(FS),
    _interleaved(interleaved)
{
    switch(_interleaved) {
        case Interleaved_Complex:
            numberOfSamples*=2;
            break;
        case Only_Real:
            break;
        default:
            throw invalid_argument( string( __FUNCTION__ ));
    }

    waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent( numberOfSamples, 1, 1)));
}


pBuffer Buffer::getInterleaved(Interleaved value) const
{
    pBuffer chunk( new Buffer( value ));
    chunk->sample_rate = sample_rate;
    chunk->sample_offset = sample_offset;

    if (value == _interleaved) {
        chunk->waveform_data.reset( new GpuCpuData<float>(waveform_data->getCpuMemory(), waveform_data->getNumberOfElements() ) );
        return chunk;
    }

    cudaExtent orgSz = waveform_data->getNumberOfElements();

    //makeCudaExtent(m*numberOfSamples, 1, 1)
    switch(value) {
        case Only_Real: {
            cudaExtent realSz = orgSz;
            realSz.width/=2;
            chunk->waveform_data.reset( new GpuCpuData<float>(0, realSz ) );

            float *complex = waveform_data->getCpuMemory();
            float *real = chunk->waveform_data->getCpuMemory();

            for (unsigned z=0; z<realSz.depth; z++)
                for (unsigned y=0; y<realSz.height; y++)
                    for (unsigned x=0; x<realSz.width; x++)
                        real[ x + (y + z*realSz.height)*realSz.width ]
                                = complex[ 2*x + (y + z*orgSz.height)*orgSz.width ];
            break;
        }
        case Interleaved_Complex: {
            cudaExtent complexSz = orgSz;
            complexSz.width*=2;
            chunk->waveform_data.reset( new GpuCpuData<float>(0, complexSz ) );

            float *complex = chunk->waveform_data->getCpuMemory();
            float *real = waveform_data->getCpuMemory();

            for (unsigned z=0; z<orgSz.depth; z++)
                for (unsigned y=0; y<orgSz.height; y++)
                    for (unsigned x=0; x<orgSz.width; x++)
                    {
                        complex[ 2*x + (y + z*complexSz.height)*complexSz.width ]
                                = real[ x + (y + z*orgSz.height)*orgSz.width ];
                        complex[ 2*x + 1 + (y + z*complexSz.height)*complexSz.width ] = 0;
                    }
            break;
        }
        default:
            throw invalid_argument( string(__FUNCTION__));
    }

    return chunk;
}

Buffer& Buffer::
        operator|=(const Buffer& rb)
{
    pBuffer b2;
    const Buffer *p = &rb;
    
    if (_interleaved != p->interleaved()) {
        b2 = p->getInterleaved(_interleaved);
        p = &*b2;
    }

    float* c = waveform_data->getCpuMemory();

    SamplesIntervalDescriptor sid( sample_offset, sample_offset + number_of_samples());
    SamplesIntervalDescriptor psid( p->sample_offset, p->sample_offset + p->number_of_samples());
    sid &= psid;

    if (sid.isEmpty())
        return *this;

    SamplesIntervalDescriptor::Interval i = sid.getInterval(p->number_of_samples());

    unsigned offs_write = i.first - sample_offset;
    unsigned offs_read = i.first - p->sample_offset;
    unsigned f = _interleaved == Interleaved_Complex ? 2: 1;
    memcpy(c+offs_write, p->waveform_data->getCpuMemory() + offs_read, f*(i.last-i.first)*sizeof(float));

    return *this;
}

pBuffer Source::
        readChecked( unsigned firstSample, unsigned numberOfSamples )
{
    pBuffer r = read(firstSample, numberOfSamples);

    if (r->sample_offset > firstSample)
        throw std::runtime_error("read didn't contain firstSample, r->sample_offset > firstSample");

    if (r->sample_offset + r->number_of_samples() <= firstSample)
        throw std::runtime_error("read didn't contain firstSample, r->sample_offset + r->number_of_samples() <= firstSample");

    return r;
}

pBuffer Source::
        readFixedLength( unsigned firstSample, unsigned numberOfSamples )
{
    // Try a simple read
    pBuffer p = readChecked(firstSample, numberOfSamples );
    if (p->number_of_samples() == numberOfSamples && p->sample_offset==firstSample)
        return p;

    // Didn't get exact result, prepare new Buffer
    pBuffer r( new Buffer(firstSample, numberOfSamples, p->sample_rate) );
    SamplesIntervalDescriptor sid(firstSample, firstSample + numberOfSamples);

    while (!sid.isEmpty())
    {
        (*r) |= *p;
        sid -= SamplesIntervalDescriptor( p->sample_offset, p->sample_offset + p->number_of_samples() );

        if (!sid.isEmpty()) {
            SamplesIntervalDescriptor::Interval i = sid.getInterval( SamplesIntervalDescriptor::SampleType_MAX );
            p = readChecked( i.first, i.last - i.first );
        }
    }

    return r;
}

} // namespace Signal