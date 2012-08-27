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


Buffer::Buffer(UnsignedF first_sample, IntervalType numberOfSamples, float fs, unsigned numberOfChannels, unsigned numberOfSignals)
:   sample_offset(first_sample),
    sample_rate(fs),
    bitor_channel_(0)
{
    BOOST_ASSERT( 0 < numberOfSamples );
    BOOST_ASSERT( 0 < numberOfChannels );
    BOOST_ASSERT( 0 < fs );
    waveform_data_.reset( new DataStorage<float>(DataStorageSize( numberOfSamples, numberOfChannels, numberOfSignals )));
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

    waveform_data_ .reset( new DataStorage<float>(subinterval.count()));
    bitor_channel_ = channel;
    *this |= *other_;
    other_.reset(); // TODO other_ was used before but isn't anymore
}


Buffer::
        ~Buffer()
{
}


DataStorage<float>::Ptr Buffer::
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
    return (sample_offset/sample_rate).asFloat();
}


float Buffer::
        length() const
{
    return number_of_samples()/sample_rate;
}


Interval Buffer::
        getInterval() const
{
    return Interval(sample_offset.asInteger(), (sample_offset + number_of_samples()).asInteger());
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

    unsigned offs_write = i.first - sample_offset.asInteger();
    unsigned offs_read = i.first - b.sample_offset.asInteger();

    TIME_BUFFER TaskTimer tt("%s %s = %s & %s",
                  __FUNCTION__ ,
                  i.toString().c_str(), getInterval().toString().c_str(), b.getInterval().toString().c_str() );

    if (bitor_channel_)
    {
        offs_read += bitor_channel_*b.number_of_samples();
        bitor_channel_ = 0;
    }

    DataStorage<float>::Ptr write, read;

    if (i == getInterval() && 1 == channels())
        write = waveform_data_;
    if (i == b.getInterval() && 1 == b.channels())
        read = b.waveform_data();

    bool toCpu = waveform_data_->HasValidContent<CpuMemoryStorage>();
    bool fromCpu = b.waveform_data_->HasValidContent<CpuMemoryStorage>();
    bool toGpu = false;
    bool fromGpu = false;

#ifdef USE_CUDA
    toGpu = waveform_data_->HasValidContent<CudaGlobalStorage>();
    fromGpu = b.waveform_data_->HasValidContent<CudaGlobalStorage>();
#endif

    if (!toCpu && !toGpu && !fromCpu && !fromGpu)
    {
        // no data was read (all 0) and no data to overwrite with 0
        return *this;
    }

#ifdef USE_CUDA
    // if no data is allocated in *this, take the gpu if 'b' has gpu storage
    if (!toCpu && !toGpu && !write)
    {
        toGpu = fromGpu;
        if (fromGpu)
            write = CudaGlobalStorage::BorrowPitchedPtr<float>(
                DataStorageSize(i.count()),
                make_cudaPitchedPtr(
                                CudaGlobalStorage::ReadWrite<1>( waveform_data_ ).device_ptr() + offs_write,
                                i.count()*sizeof(float),
                                i.count()*sizeof(float), 1), false);
        else
            write = CpuMemoryStorage::BorrowPtr(
                DataStorageSize(i.count()),
                CpuMemoryStorage::ReadWrite<1>( waveform_data_ ).ptr() + offs_write, false);
    }

    if (!fromCpu && !fromGpu)
        fromGpu = toGpu;

    if (toGpu || fromGpu)
    {
        if (toGpu && !write)
            write = CudaGlobalStorage::BorrowPitchedPtr<float>(
                DataStorageSize(i.count()),
                make_cudaPitchedPtr(
                                CudaGlobalStorage::ReadWrite<1>( waveform_data_ ).device_ptr() + offs_write,
                                i.count()*sizeof(float),
                                i.count()*sizeof(float), 1), false);


        if (fromGpu && !read)
            read = CudaGlobalStorage::BorrowPitchedPtr<float>(
                DataStorageSize(i.count()),
                make_cudaPitchedPtr(
                        CudaGlobalStorage::ReadOnly<1>( b.waveform_data_ ).device_ptr() + offs_read,
                        i.count()*sizeof(float),
                        i.count()*sizeof(float), 1), false);
    }
#endif

    if (!write)
        write = CpuMemoryStorage::BorrowPtr(
            DataStorageSize(i.count()),
            CpuMemoryStorage::ReadWrite<1>( waveform_data_ ).ptr() + offs_write, false);

    if (!read)
        read = CpuMemoryStorage::BorrowPtr(
            DataStorageSize(i.count()),
            CpuMemoryStorage::ReadOnly<1>( b.waveform_data_ ).ptr() + offs_read, false);

    // Let DataStorage manage all memcpying
    TIME_BUFFER_LINE( *write = *read );

    return *this;
}



Buffer& Buffer::
        operator+=(const Buffer& b)
{
    Interval i = getInterval() & b.getInterval();

    if (0 == i.count())
        return *this;

    unsigned offs_write = i.first - sample_offset.asInteger();
    unsigned offs_read = i.first - b.sample_offset.asInteger();
    unsigned length = i.count();

    float* write = waveform_data_->getCpuMemory();
    float const* read = b.waveform_data_->getCpuMemory();

    write += offs_write;
    read += offs_read;

    for (unsigned n=0; n<length; n++)
        write[n] += read[n];

    return *this;
}


} // namespace Signal
