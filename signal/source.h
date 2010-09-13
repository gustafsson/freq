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
    enum Interleaved {
        Interleaved_Complex,
        Only_Real
    };

    Buffer(Interleaved interleaved = Only_Real);
    Buffer(unsigned firstSample,
           unsigned numberOfSamples,
           unsigned FS,
           Interleaved interleaved = Only_Real
                                   );

    boost::scoped_ptr<GpuCpuData<float> >
                    waveform_data;

    unsigned        sample_offset;
    unsigned        sample_rate;

    Interleaved     interleaved() const { return _interleaved; }
    unsigned        number_of_samples() const { return waveform_data->getNumberOfElements().width/(_interleaved==Interleaved_Complex?2:1); }
    float           start() const { return sample_offset/(float)sample_rate; }
    float           length() const { return number_of_samples()/(float)sample_rate; }
    Interval        getInterval() const { return Interval(sample_offset, sample_offset + number_of_samples()); }

    Buffer&         operator|=(const Buffer& b);

    boost::shared_ptr<Buffer>
                    getInterleaved(Interleaved) const;

private:
    const Interleaved _interleaved;
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
      to start at I.first. The caller of read must allow for read to return
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
};


} // namespace Signal

#endif // SIGNALSOURCE_H
