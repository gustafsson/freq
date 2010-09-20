#ifndef SIGNALSOURCE_H
#define SIGNALSOURCE_H

#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <GpuCpuData.h>
#include "signal/intervals.h"

namespace Signal {

/**
The Signal::Buffer class is _the_ container class for data in the Signal 
namespace. A Buffer can contain an entire song as when created by
Signal::Audiofile, or a Buffer can contain sound as fractions of a second
as when created by Signal::MicrophoneRecorder.
*/
class Buffer {
public:
    Buffer(unsigned firstSample,
           unsigned numberOfSamples,
           unsigned FS,
           unsigned numberOfChannels=1);

    GpuCpuData<float>*  waveform_data() const;
    unsigned            number_of_samples() const;
    void                release_extra_resources();

    unsigned        sample_offset;
    unsigned        sample_rate;

    float           start() const;
    float           length() const;
    Interval        getInterval() const;

    Buffer&         operator|=(const Buffer& b);

protected:
    boost::scoped_ptr<GpuCpuData<float> >
                    _waveform_data;
};
typedef boost::shared_ptr<Buffer> pBuffer;


/**
Signal::Source is a very important class in the Signal namespace. It declares 
an interface through which buffers can be read.
*/
class SourceBase
{
public:
    virtual ~SourceBase() {}

    /**
      read does not have to return a Buffer of the same size as I. But it has
      to include I.first. The caller of read must allow for read to return
      Buffers of arbitrary sizes.

      However, read shall never return pBuffer(). Either throw an logic_error()
      exception or create a new Buffer with zeros.
    */
    virtual pBuffer read( const Interval& I ) = 0;
    virtual pBuffer readChecked( const Interval& I );
    virtual pBuffer readFixedLength( const Interval& I );
    virtual unsigned sample_rate() = 0;
    virtual long unsigned number_of_samples() = 0;

    float length() { return number_of_samples() / (float)sample_rate(); }

protected:
    virtual pBuffer zeros( const Interval& I );
};


} // namespace Signal

#endif // SIGNALSOURCE_H
