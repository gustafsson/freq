#ifndef SIGNALSOURCE_H
#define SIGNALSOURCE_H

#include "intervals.h"

// gpumisc
#include <GpuCpuData.h>
#include <unsignedf.h>

// boost
#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>

namespace Signal {


/**
The Signal::Buffer class is _the_ container class for data in the Signal 
namespace. A Buffer can contain an entire song as when created by
Signal::Audiofile, or a Buffer can contain sound as fractions of a second
as when created by Signal::MicrophoneRecorder.
*/
class Buffer : public boost::noncopyable {
public:
    Buffer(UnsignedF firstSample,
           IntervalType numberOfSamples,
           float sample_rate,
           unsigned numberOfChannels = 1);
    ~Buffer();

    GpuCpuData<float>*  waveform_data() const;
    IntervalType        number_of_samples() const
    {
        return waveform_data_->getNumberOfElements().width;
    }
    void                release_extra_resources();

    UnsignedF       sample_offset;
    float           sample_rate;

    float           start() const;
    float           length() const;
    Interval        getInterval() const;

    /// element-wise overwrite 'this' with data from 'b' where they overlap
    Buffer&         operator|=(const Buffer& b);
    /// element-wise add 'this' with 'b' where they overlap
    Buffer&         operator+=(const Buffer& b);

protected:
    GpuCpuData<float> *waveform_data_;
};
typedef boost::shared_ptr<Buffer> pBuffer;


/**
Signal::Source is the single most important class in the Signal namespace. It
declares an interface through which buffers can be read.
*/
class SourceBase
{
public:
    /**
      Virtual housekeeping.
     */
    virtual ~SourceBase() {}


    /**
      read does not have to return a Buffer of the same size as I. But it has
      to include I.first. The caller of read must allow for read to return
      Buffers of arbitrary sizes.

      However, read shall never return a null pBuffer(). Either throw an
      logic_error() exception or create a new Buffer with zeros.
    */
    virtual pBuffer read( const Interval& I ) = 0;
    virtual float sample_rate() = 0;
    virtual IntervalType number_of_samples() = 0;


    /**
      Creates a buffer of the exact interval 'I'. Takes data from 'read' and
      crops it if necessary.
      */
    pBuffer readFixedLength( const Interval& I );


    /**
      Length of signal, in seconds.
      */
    float length() { return number_of_samples() / sample_rate(); }
    std::string lengthLongFormat() { return lengthLongFormat(length()); }
    static std::string lengthLongFormat( float T );


protected:
    /**
      Used by 'readFixedLength' and others to create a buffer with zeros. The
      buffer will be over the exact interval 'I'.
      */
    pBuffer zeros( const Interval& I );


private:
    /**
      Calls 'read' and checks that it returns valid data. That is; a non-empty
      buffer containing I.first. Used by 'readFixedLength'.
      */
    pBuffer readChecked( const Interval& I );
};


} // namespace Signal

#endif // SIGNALSOURCE_H
