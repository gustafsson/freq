#include "source.h"

#include <string.h>
#include <stdio.h>
#include <demangle.h>
#include <typeinfo>

#include <sstream>

#include "cpumemorystorage.h"
#include "TaskTimer.h"

//#define TIME_READCHECKED
#define TIME_READCHECKED if(0)

using namespace std;

namespace Signal {


Buffer::Buffer(UnsignedF first_sample, IntervalType numberOfSamples, float fs, unsigned numberOfChannels, unsigned numberOfSignals)
:   sample_offset(first_sample),
    sample_rate(fs),
    bitor_channel_(0)
{
    BOOST_ASSERT( 0 < numberOfSamples );
    BOOST_ASSERT( 0 < numberOfChannels );
    BOOST_ASSERT( 0 < fs );
    waveform_data_.reset( new DataStorage<float, 3>(DataStorageSize( numberOfSamples, numberOfChannels, numberOfSignals )));
}


Buffer::Buffer(Signal::Interval subinterval, pBuffer other, unsigned channel )
:   sample_offset(subinterval.first),
    sample_rate(other->sample_rate),
    other_(other)
{
    while (other->other_ && other.get() != this) 
        other = other->other_;

    BOOST_ASSERT( 0 < sample_rate );
    BOOST_ASSERT( (subinterval & other->getInterval()) == subinterval );
    BOOST_ASSERT( other.get() != this );
    BOOST_ASSERT( channel < other->channels() );

/*    DataStorage<float, 3>& data = *other_->waveform_data();
    
    IntervalType offs = channel*other_->number_of_samples() + subinterval.first - other->getInterval().first;

    if(0) switch (data.getMemoryLocation())
    {
    case GpuCpuVoidData::CpuMemory:
        waveform_data_ = new GpuCpuData<float>(
                data.getCpuMemory() + offs,
                make_uint3( subinterval.count(), 1, 1), GpuCpuVoidData::CpuMemory, true );
        return;

    case GpuCpuVoidData::CudaGlobal:
        {
            cudaPitchedPtrType<float> cppt = data.getCudaGlobal();
            cudaPitchedPtr cpp = cppt.getCudaPitchedPtr();
            cpp.ptr = ((float*)cpp.ptr) + offs;
            cpp.xsize = sizeof(float) * subinterval.count();
            cpp.ysize = 1;
            cpp.pitch = cpp.xsize;
            cppt = cudaPitchedPtrType<float>( cpp );

            waveform_data_ = new GpuCpuData<float>(
                    &cpp,
                    cppt.getNumberOfElements(),
                    GpuCpuVoidData::CudaGlobal, true );
        }
        return;

    default:
        break;
    }*/

    waveform_data_ .reset( new DataStorage<float, 3>(subinterval.count()));
    bitor_channel_ = channel;
    *this |= *other_;
    other_.reset();
}


Buffer::
        ~Buffer()
{
}


DataStorage<float, 3>::Ptr Buffer::
        waveform_data() const
{
    return waveform_data_;
}


void Buffer::
        release_extra_resources()
{
    waveform_data_->OnlyKeepOneStorage<CpuMemoryStorage>();
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


unsigned Buffer::
        channels() const
{
    return waveform_data()->size().height;
}


Buffer& Buffer::
        operator|=(const Buffer& b)
{    
    Interval i = getInterval() & b.getInterval();

    if (0 == i.count())
        return *this;

    unsigned offs_write = i.first - sample_offset;
    unsigned offs_read = i.first - b.sample_offset;

    if (bitor_channel_)
    {
        offs_read += bitor_channel_*b.number_of_samples();
        bitor_channel_ = 0;
    }

    float* write;
    float const* read;

    write = &CpuMemoryStorage::ReadWrite( waveform_data_ ).ref( offs_write );
    read = &CpuMemoryStorage::ReadOnly( b.waveform_data_ ).ref( offs_read );

    memcpy(write, read, i.count()*sizeof(float));

    /*
    bool toGpu   = waveform_data_->getMemoryLocation() == GpuCpuVoidData::CudaGlobal;
    bool fromGpu = b.waveform_data_->getMemoryLocation() == GpuCpuVoidData::CudaGlobal;

    if ( toGpu )    write = waveform_data_->getCudaGlobal().ptr();
    else            write = waveform_data_->getCpuMemory();

    if ( fromGpu )  read = b.waveform_data_->getCudaGlobal().ptr();
    else            read = b.waveform_data_->getCpuMemory();

    write += offs_write;
    read += offs_read;

    cudaMemcpyKind kind = (cudaMemcpyKind)(1*toGpu | 2*fromGpu);
    if (!toGpu && !fromGpu)
        memcpy(write, read, i.count()*sizeof(float));
    else
        cudaMemcpy(write, read, i.count()*sizeof(float), kind );
    */

    return *this;
}



Buffer& Buffer::
        operator+=(const Buffer& b)
{
    Interval i = getInterval() & b.getInterval();

    if (0 == i.count())
        return *this;

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

    // Check if read returned any samples form the interval I
    BOOST_ASSERT(r->sample_offset < I.last);
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

    // This row gives some performance gain (cpu->gpu copy only once and never back until inverse).
    // But this also increases complexity to be handled properyl, that is not coded yet.
    //p->waveform_data()->getCudaGlobal();

    // Didn't get exact result, check if it spans all of I
    if ( (p->getInterval() & I) == I )
    {
        pBuffer r( new Buffer( I, p ));
        BOOST_ASSERT( r->getInterval() == I );
        return r;
    }

    // Doesn't span all of I, prepare new Buffer
    pBuffer r( new Buffer(I.first, I.count(), p->sample_rate ) );

//    if ( p->waveform_data()->getMemoryLocation() == GpuCpuVoidData::CudaGlobal )
//        r->waveform_data()->getCudaGlobal();

    Intervals sid(I);

    while (sid)
    {
        if (!p)
            p = readChecked( sid.fetchFirstInterval() );

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
        ss << L << " s";
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
