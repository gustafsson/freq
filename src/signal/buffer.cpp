#include "buffer.h"

#include <string.h> //memcpy

#include "TaskTimer.h"
#include "cpumemorystorage.h"
#ifdef USE_CUDA
#include "cudaglobalstorage.h"
#endif


//#define TIME_BUFFER
#define TIME_BUFFER if(0)

//#define TIME_BUFFER_LINE(x) TIME(x)
#define TIME_BUFFER_LINE(x) x


namespace Signal {


MonoBuffer::
        MonoBuffer(Interval I, float fs)
:   sample_offset_(I.first),
    sample_rate_(fs)
{
    EXCEPTION_ASSERT( 0 < I.count ());
    EXCEPTION_ASSERT( 0 < sample_rate() );

    time_series_.reset( new TimeSeriesData(DataStorageSize( I.count ())));
}


MonoBuffer::
        MonoBuffer(UnsignedF first_sample, IntervalType numberOfSamples, float fs)
:   sample_offset_(first_sample),
    sample_rate_(fs)
{
    EXCEPTION_ASSERT( 0 < numberOfSamples );
    EXCEPTION_ASSERT( 0 < sample_rate() );

    time_series_.reset( new TimeSeriesData(DataStorageSize( numberOfSamples )));
}


MonoBuffer::
        MonoBuffer(UnsignedF first_sample, pTimeSeriesData p, float fs )
:   sample_offset_(first_sample),
    sample_rate_(fs)
{
    EXCEPTION_ASSERT( 0 < sample_rate() );
    EXCEPTION_ASSERT( 1 == p->size ().height);
    EXCEPTION_ASSERT( 1 == p->size ().depth);

    time_series_.reset( new TimeSeriesData(p->size ()));

    time_series_ = p;
}


MonoBuffer::
        ~MonoBuffer()
{
}


void MonoBuffer::
        release_extra_resources()
{
    time_series_->OnlyKeepOneStorage<CpuMemoryStorage>();
}


float MonoBuffer::
        start() const
{
    return (sample_offset_/sample_rate_).asFloat();
}


float MonoBuffer::
        length() const
{
    return number_of_samples()/sample_rate_;
}


Interval MonoBuffer::
        getInterval() const
{
    return Interval(sample_offset_.asInteger(), (sample_offset_ + number_of_samples()).asInteger());
}


MonoBuffer& MonoBuffer::
        operator|=(const MonoBuffer& b)
{
    Interval i = getInterval() & b.getInterval();

    if (0 == i.count())
        return *this;

    unsigned offs_write = i.first - sample_offset().asInteger();
    unsigned offs_read = i.first - b.sample_offset().asInteger();

    TIME_BUFFER TaskTimer tt("%s %s = %s & %s",
                  __FUNCTION__ ,
                  i.toString().c_str(), getInterval().toString().c_str(), b.getInterval().toString().c_str() );

    bool toCpu = time_series_->HasValidContent<CpuMemoryStorage>();
    bool fromCpu = b.time_series_->HasValidContent<CpuMemoryStorage>();
    bool toGpu = false;
    bool fromGpu = false;

#ifdef USE_CUDA
    toGpu = time_series_->HasValidContent<CudaGlobalStorage>();
    fromGpu = b.time_series_->HasValidContent<CudaGlobalStorage>();

    // if no data is allocated in *this, take the gpu if 'b' has gpu storage
    if (!toCpu && !toGpu)
        toGpu = fromGpu;

    if (!fromCpu && !fromGpu)
        fromGpu = toGpu;
#endif

    if (!toCpu && !toGpu && !fromCpu && !fromGpu)
    {
        // no data was read (all 0) and no data to overwrite with 0
        return *this;
    }

    pTimeSeriesData write, read;

    if (i == getInterval())
    {
        write = time_series_;
#ifdef USE_CUDA
        if (toGpu)
            CudaGlobalStorage::WriteAll<1>(write);
        else
#endif
            CpuMemoryStorage::WriteAll<1>(write);
    }
    if (i == b.getInterval())
    {
        read = b.waveform_data();
#ifdef USE_CUDA
        if (fromGpu)
            CudaGlobalStorage::ReadOnly<1>(read);
        else
#endif
            CpuMemoryStorage::ReadOnly<1>(read);
    }

#ifdef USE_CUDA
    if (toGpu && !write)
        write = CudaGlobalStorage::BorrowPitchedPtr<float>(
            DataStorageSize(i.count()),
            make_cudaPitchedPtr(
                            CudaGlobalStorage::ReadWrite<1>( time_series_ ).device_ptr() + offs_write,
                            i.count()*sizeof(float),
                            i.count()*sizeof(float), 1), false);


    if (fromGpu && !read)
        read = CudaGlobalStorage::BorrowPitchedPtr<float>(
            DataStorageSize(i.count()),
            make_cudaPitchedPtr(
                    CudaGlobalStorage::ReadOnly<1>( b.time_series_ ).device_ptr() + offs_read,
                    i.count()*sizeof(float),
                    i.count()*sizeof(float), 1), false);
#endif

    if (!write)
        write = CpuMemoryStorage::BorrowPtr(
            DataStorageSize(i.count()),
            CpuMemoryStorage::ReadWrite<1>( time_series_ ).ptr() + offs_write, false);

    if (!read)
        read = CpuMemoryStorage::BorrowPtr(
            DataStorageSize(i.count()),
            CpuMemoryStorage::ReadOnly<1>( b.time_series_ ).ptr() + offs_read, false);

    // Let DataStorage manage all memcpying
    TIME_BUFFER_LINE( *write = *read );

    return *this;
}



MonoBuffer& MonoBuffer::
        operator+=(const MonoBuffer& b)
{
    Interval i = getInterval() & b.getInterval();

    if (0 == i.count())
        return *this;

    unsigned offs_write = i.first - sample_offset().asInteger();
    unsigned offs_read = i.first - b.sample_offset().asInteger();
    unsigned length = i.count();

    float* write = time_series_->getCpuMemory();
    float const* read = b.time_series_->getCpuMemory();

    write += offs_write;
    read += offs_read;

    for (unsigned n=0; n<length; n++)
        write[n] += read[n];

    return *this;
}


bool MonoBuffer::
        operator==(MonoBuffer const& b) const
{
    if (b.waveform_data ()->size () != waveform_data ()->size ())
        return false;
    float *p = waveform_data ()->getCpuMemory ();
    float *bp = b.waveform_data ()->getCpuMemory ();
    return 0 == memcmp(p, bp, waveform_data ()->numberOfBytes ());
}


Buffer::
        Buffer(Interval I,
       float sample_rate,
       unsigned number_of_channels)
{
    EXCEPTION_ASSERT( 0 < sample_rate );
    EXCEPTION_ASSERT( 0 < number_of_channels );

    channels_.resize(number_of_channels);
    for (unsigned i=0; i<number_of_channels; ++i)
        channels_[i].reset(new MonoBuffer(I, sample_rate));
}


Buffer::
        Buffer(UnsignedF first_sample,
       IntervalType number_of_samples,
       float sample_rate,
       unsigned number_of_channels)
{
    EXCEPTION_ASSERT( 0 < sample_rate );
    EXCEPTION_ASSERT( 0 < number_of_channels );

    channels_.resize(number_of_channels);
    for (unsigned i=0; i<number_of_channels; ++i)
        channels_[i].reset(new MonoBuffer(first_sample, number_of_samples, sample_rate));
}


Buffer::
        Buffer(pMonoBuffer b)
{
    channels_.resize (1);
    channels_[0] = b;
}


Buffer::
        Buffer(UnsignedF first_sample, pTimeSeriesData ptr, float sample_rate)
{
    DataStorageSize sz = ptr->size ();
    EXCEPTION_ASSERT( 1 == sz.depth );
    channels_.resize (sz.height);

    float* p = ptr->getCpuMemory ();
    for (unsigned i=0; i<number_of_channels (); ++i)
    {
        pTimeSeriesData qtr(new TimeSeriesData(sz.width));
        float* q = qtr->getCpuMemory();
        memcpy(q, p + i*sz.width, sz.width*sizeof(float));
        channels_[i].reset(new MonoBuffer(first_sample, qtr, sample_rate));
    }
}


Buffer::
        ~Buffer()
{}


void Buffer::
        release_extra_resources()
{
    for (unsigned i=0; i<number_of_channels(); ++i)
        channels_[i]->release_extra_resources();
}


void Buffer::
        set_sample_rate(float fs)
{
    for (unsigned i=0; i<number_of_channels(); ++i)
        channels_[i]->set_sample_rate(fs);
}


void Buffer::
        set_sample_offset(UnsignedF offset)
{
    for (unsigned i=0; i<number_of_channels(); ++i)
        channels_[i]->set_sample_offset(offset);
}


pTimeSeriesData Buffer::
        mergeChannelData() const
{
    if (1 == number_of_channels())
        return getChannel (0)->waveform_data ();

    pTimeSeriesData r( new TimeSeriesData(number_of_samples(), number_of_channels()));
    float* p = r->getCpuMemory ();
    for (unsigned i=0; i<number_of_channels(); ++i)
    {
        float* q = getChannel(i)->waveform_data ()->getCpuMemory ();
        memcpy(p + i*number_of_samples (), q, number_of_samples()*sizeof(float));
    }
    return r;
}


Buffer& Buffer::
        operator|=(const Buffer& b)
{
    EXCEPTION_ASSERT( b.number_of_channels () == number_of_channels ());
    for (unsigned i=0; i<number_of_channels(); ++i)
        *channels_[i] |= *b.getChannel (i);
    return *this;
}


Buffer& Buffer::
        operator+=(const Buffer& b)
{
    EXCEPTION_ASSERT( b.number_of_channels () == number_of_channels ());
    for (unsigned i=0; i<number_of_channels(); ++i)
        *channels_[i] += *b.getChannel (i);
    return *this;
}


bool Buffer::
        operator==(const Buffer& b) const
{
    if (b.number_of_channels () != number_of_channels ())
        return false;

    for (unsigned i=0; i<number_of_channels(); ++i)
        if (*channels_[i] != *b.getChannel (i))
            return false;

    return true;
}


void Buffer::
        test()
{
    pBuffer b(new Buffer(Interval(20,30), 40, 7));
    for (unsigned c=0; c<b->number_of_channels (); ++c)
    {
        float *p = b->getChannel (c)->waveform_data ()->getCpuMemory ();
        for (int i=0; i<b->number_of_samples (); ++i)
            p[i] = c + i/(float)b->number_of_samples ();
    }
    pBuffer c(new Buffer(Interval(20,30), 40, 7));
    *c |= *b;

    // Test that 'c' contains a copy on write of 'b'
    float * bp = CpuMemoryStorage::ReadOnly<1>(b->getChannel (0)->waveform_data ()).ptr();
    float * cp = CpuMemoryStorage::ReadOnly<1>(c->getChannel (0)->waveform_data ()).ptr();
    float * bpcpu = b->getChannel (0)->waveform_data ()->getCpuMemory ();
    float * cp2 = CpuMemoryStorage::ReadOnly<1>(c->getChannel (0)->waveform_data ()).ptr();
    float * cpcpu = c->getChannel (0)->waveform_data ()->getCpuMemory ();
    EXCEPTION_ASSERTX( bp != 0, "Buffer didn't allocate any data");
    EXCEPTION_ASSERTX( bp == cp, "Buffer |= didn't do a copy on write");
    EXCEPTION_ASSERTX( bpcpu == bp, "Buffer |= didn't do a copy on write");
    EXCEPTION_ASSERTX( bpcpu != cp2, "Buffer |= didn't do a copy on write");
    EXCEPTION_ASSERTX( cpcpu == cp2, "Buffer |= didn't do a copy on write");
}

} // namespace Signal
