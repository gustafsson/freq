#include "source.h"

#include <string.h>
#include <stdio.h>
#include <demangle.h>
#include <typeinfo>

//#define TIME_READCHECKED
#define TIME_READCHECKED if(0)

using namespace std;

namespace Signal {


Buffer::Buffer(UnsignedF first_sample, IntervalType numberOfSamples, float fs, unsigned numberOfChannels)
:   sample_offset(first_sample),
    sample_rate(fs)
{
    if (numberOfSamples)
        _waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent( numberOfSamples, numberOfChannels, 1)));
}


GpuCpuData<float>* Buffer::
        waveform_data() const
{
    return _waveform_data.get();
}


long unsigned Buffer::
        number_of_samples() const
{
    return _waveform_data->getNumberOfElements().width;
}


void Buffer::
        release_extra_resources()
{
    _waveform_data->getCpuMemory();
    _waveform_data->freeUnused();
}


float Buffer::
        start() const
{
    return sample_offset/sample_rate;
}


float Buffer::
        length() const
{
    return number_of_samples()/sample_rate;
}


Interval Buffer::
        getInterval() const
{
    return Interval(sample_offset, sample_offset + number_of_samples());
}


Buffer& Buffer::
        operator|=(const Buffer& b)
{    
    Intervals sid = getInterval();
    sid &= b.getInterval();

    if (sid.empty())
        return *this;

    Interval i = sid.getInterval(b.number_of_samples());

    unsigned offs_write = i.first - sample_offset;
    unsigned offs_read = i.first - b.sample_offset;

    float* write = waveform_data()->getCpuMemory();
    float* read = b.waveform_data()->getCpuMemory();

    write+=offs_write;
    read+=offs_read;

    memcpy(write, read, i.count()*sizeof(float));

    return *this;
}



Buffer& Buffer::
        operator+=(const Buffer& b)
{
    Intervals sid = getInterval();
    sid &= b.getInterval();

    if (sid.empty())
        return *this;

    Interval i = sid.getInterval(b.number_of_samples());

    unsigned offs_write = i.first - sample_offset;
    unsigned offs_read = i.first - b.sample_offset;
    unsigned length = i.count();

    float* write = waveform_data()->getCpuMemory();
    float* read = b.waveform_data()->getCpuMemory();

    write+=offs_write;
    read+=offs_read;

    for (unsigned n=0; n<length; n++)
        write[n] += read[n];

    return *this;
}


pBuffer SourceBase::
        readChecked( const Interval& I )
{
    BOOST_ASSERT( I.valid() );

    pBuffer r = read(I);

    // Check if read contains firstSample
    BOOST_ASSERT(r->sample_offset <= I.first);
    BOOST_ASSERT(r->sample_offset + r->number_of_samples() > I.first);

    return r;
}

pBuffer SourceBase::
        readFixedLength( const Interval& I )
{
    std::stringstream ss;
    TIME_READCHECKED ss << I;
    TIME_READCHECKED TaskTimer tt("%s.%s %s",
                  demangle(typeid(*this).name()).c_str(), __FUNCTION__ ,
                  ss.str().c_str() );

    // Try a simple read
    pBuffer p = readChecked( I );
    if (I == p->getInterval())
        return p;

    // Didn't get exact result, prepare new Buffer
    pBuffer r( new Buffer(I.first, I.count(), sample_rate()) );

    Intervals sid(I);

    while (sid)
    {
        if (!p)
            p = readChecked( sid.getInterval() );

        sid -= p->getInterval();
        (*r) |= *p; // Fill buffer
        p.reset();
    }

    return r;
}

pBuffer SourceBase::
        zeros( const Interval& I )
{
    BOOST_ASSERT( I.valid() );
    std::stringstream ss;
    TIME_READCHECKED ss << I;
    TIME_READCHECKED TaskTimer tt("%s.%s %s",
                  demangle(typeid(*this).name()).c_str(), __FUNCTION__ ,
                  ss.str().c_str() );

    pBuffer r( new Buffer(I.first, I.count(), sample_rate()) );
    memset(r->waveform_data()->getCpuMemory(), 0, r->waveform_data()->getSizeInBytes1D());
    return r;
}


} // namespace Signal
