#ifndef SIGNALSOURCE_H
#define SIGNALSOURCE_H

#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <GpuCpuData.h>
#include <set>

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

    Buffer(Interleaved interleaved=Only_Real);
    Buffer(unsigned firstSample, unsigned numberOfSamples, unsigned FS, Interleaved interleaved=Only_Real);

    boost::scoped_ptr<GpuCpuData<float> > waveform_data;

    unsigned number_of_samples() const { return waveform_data->getNumberOfElements().width/(_interleaved==Interleaved_Complex?2:1); }
    Interleaved interleaved() const {return _interleaved; }
    boost::shared_ptr<class Buffer> getInterleaved(Interleaved) const;

    float start() { return sample_offset/(float)sample_rate; }
    float length() { return number_of_samples()/(float)sample_rate; }

    Buffer& operator|=(const Buffer& b);
    unsigned sample_offset;
    unsigned sample_rate;

private:
    const Interleaved _interleaved;
};
typedef boost::shared_ptr<class Buffer> pBuffer;


/**
Signal::Source is a very important class in the Signal namespace. It declares 
an interface through which buffers can be read.
*/
class Source
{
public:
    virtual ~Source() {}

    /**
    read does not have to return a Buffer of the same size as numberOfSamples.
    But it has to start at firstSample. The caller of read must allow for read
    to return Buffers of arbitrary sizes.
    */
    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples ) = 0;
    virtual pBuffer readChecked( unsigned firstSample, unsigned numberOfSamples );
    virtual pBuffer readFixedLength( unsigned firstSample, unsigned numberOfSamples );
    virtual unsigned sample_rate() = 0;
    virtual unsigned number_of_samples() = 0;

    float length() { return number_of_samples() / (float)sample_rate(); }
};
typedef boost::shared_ptr<class Source> pSource;


} // namespace Signal

#endif // SIGNALSOURCE_H
