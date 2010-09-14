#include "source.h"

#include <string.h>
#include <stdio.h>

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
        p = b2.get();
    }

    float* c = waveform_data->getCpuMemory();

    Intervals sid = getInterval();
    sid &= p->getInterval();

    if (sid.isEmpty())
        return *this;

    Interval i = sid.getInterval(p->number_of_samples());

    unsigned offs_write = i.first - sample_offset;
    unsigned offs_read = i.first - p->sample_offset;
    unsigned f = ((_interleaved == Interleaved_Complex) ? 2: 1);
    memcpy(c+f*offs_write, p->waveform_data->getCpuMemory() + f*offs_read, f*(i.last-i.first)*sizeof(float));

    return *this;
}

pBuffer SourceBase::
        readChecked( const Interval& I )
{
    pBuffer r = read(I);

    printf("r->sample_offset = %u\n", r->sample_offset);
    printf("r->number_of_samples() = %u\n", r->number_of_samples());

    // Check if read contains firstSample
    BOOST_ASSERT(r->sample_offset <= I.first);
    BOOST_ASSERT(r->sample_offset + r->number_of_samples() > I.first);

    return r;
}

pBuffer SourceBase::
        readFixedLength( const Interval& I )
{
    // Try a simple read
    pBuffer p = readChecked( I );
    if (p->number_of_samples() == I.count() && p->sample_offset==I.first)
        return p;

    // Didn't get exact result, prepare new Buffer
    pBuffer r = zeros(I);

    Intervals sid(I);

    while (!sid.isEmpty())
    {
        if (!p)
            p = readChecked( sid.getInterval() );

        sid -= p->getInterval();
        (*r) |= *p;
        p.reset();
    }

    return r;
}

pBuffer SourceBase::
        zeros( const Interval& I )
{
    pBuffer r( new Buffer(I.first, I.count(), sample_rate()) );
    memset(r->waveform_data->getCpuMemory(), 0, r->waveform_data->getSizeInBytes1D());
    return r;
}


} // namespace Signal
