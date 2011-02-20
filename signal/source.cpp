#include "source.h"

#include <string.h>
#include <stdio.h>
#include <demangle.h>
#include <typeinfo>

#include <sstream>

//#define TIME_READCHECKED
#define TIME_READCHECKED if(0)

using namespace std;

namespace Signal {


Buffer::Buffer(UnsignedF first_sample, IntervalType numberOfSamples, float fs, unsigned numberOfChannels)
:   sample_offset(first_sample),
    sample_rate(fs)
{
    BOOST_ASSERT( 0 < numberOfSamples );
    BOOST_ASSERT( 0 < numberOfChannels );
    BOOST_ASSERT( 0 < fs );
    waveform_data_ = new GpuCpuData<float>(0, make_cudaExtent( numberOfSamples, numberOfChannels, 1));
}


Buffer::
        ~Buffer()
{
    delete waveform_data_;
}


GpuCpuData<float>* Buffer::
        waveform_data() const
{
    return waveform_data_;
}


void Buffer::
        release_extra_resources()
{
    waveform_data_->getCpuMemory();
    waveform_data_->freeUnused();
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

    Interval i = sid.getInterval();

    unsigned offs_write = i.first - sample_offset;
    unsigned offs_read = i.first - b.sample_offset;

    float* write = waveform_data_->getCpuMemory();
    float const* read = b.waveform_data_->getCpuMemory();

    write += offs_write;
    read += offs_read;

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

    Interval i = sid.getInterval();

    unsigned offs_write = i.first - sample_offset;
    unsigned offs_read = i.first - b.sample_offset;
    unsigned length = i.count();

    float* write = waveform_data_->getCpuMemory();
    float const* read = b.waveform_data_->getCpuMemory();

    write += offs_write;
    read += offs_read;

    for (unsigned n=0; n<length; n++)
        write[n] += read[n];

    return *this;
}


pBuffer SourceBase::
        readChecked( const Interval& I )
{
    BOOST_ASSERT( I.count() );

    pBuffer r = read(I);

    // Check if read contains firstSample
    BOOST_ASSERT(r->sample_offset <= I.first);
    BOOST_ASSERT(r->sample_offset + r->number_of_samples() > I.first);

    return r;
}

pBuffer SourceBase::
        readFixedLength( const Interval& I )
{
    TIME_READCHECKED TaskTimer tt("%s.%s %s",
                  vartype(*this).c_str(), __FUNCTION__ ,
                  I.toString().c_str() );

    // Try a simple read
    pBuffer p = readChecked( I );
    if (I == p->getInterval())
        return p;

    // Didn't get exact result, prepare new Buffer
    //pBuffer r( new Buffer(I.first, I.count(), p->sample_rate ) );
    pBuffer r( new Buffer(I.first, I.count(), sample_rate() ) );

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


std::string SourceBase::
        lengthLongFormat(float L)
{
    std::stringstream ss;
    unsigned seconds_per_minute = 60;
    unsigned seconds_per_hour = seconds_per_minute*60;
    unsigned seconds_per_day = seconds_per_hour*24;

    if (L < seconds_per_minute )
        ss << L << " seconds";
    else
    {
        if (L <= seconds_per_day )
        {
            unsigned days = floor(L/seconds_per_day);
            ss << days << "d ";
            L -= days * seconds_per_day;
        }

        unsigned hours = floor(L/seconds_per_hour);
        ss << std::setfill('0') << std::setw(2) << hours << ":";
        L -= hours * seconds_per_hour;

        unsigned minutes = floor(L/seconds_per_minute);
        ss << std::setfill('0') << std::setw(2) << minutes << ":";
        L -= minutes * seconds_per_minute;

        ss << std::setiosflags(std::ios::fixed)
           << std::setprecision(3) << std::setw(6) << L;
    }
    return ss.str();
}


pBuffer SourceBase::
        zeros( const Interval& I )
{
    BOOST_ASSERT( I.count() );

    TIME_READCHECKED TaskTimer tt("%s.%s %s",
                  vartype(*this).c_str(), __FUNCTION__ ,
                  I.toString().c_str() );

    pBuffer r( new Buffer(I.first, I.count(), sample_rate()) );
    memset(r->waveform_data()->getCpuMemory(), 0, r->waveform_data()->getSizeInBytes1D());
    return r;
}


} // namespace Signal
