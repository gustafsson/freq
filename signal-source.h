#ifndef SIGNALSOURCE_H
#define SIGNALSOURCE_H

#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <GpuCpuData.h>
#include <set>

namespace Signal {

class Buffer {
public:
    enum Interleaved {
        Interleaved_Complex,
        Only_Real
    };

    Buffer(Interleaved interleaved=Only_Real);

    boost::scoped_ptr<GpuCpuData<float> > waveform_data;

    unsigned number_of_samples() { return waveform_data->getNumberOfElements().width/(_interleaved==Interleaved_Complex?2:1); }
    Interleaved interleaved() const {return _interleaved; }
    boost::shared_ptr<class Buffer> getInterleaved(Interleaved);

    float start() { return sample_offset/(float)sample_rate; }
    float length() { return number_of_samples()/(float)sample_rate; }

    unsigned sample_offset;
    unsigned sample_rate;
    bool modified;
    bool was_modified;

private:
    const Interleaved _interleaved;
};
typedef boost::shared_ptr<class Buffer> pBuffer;


class Source
{
public:
    virtual ~Source() {}

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples ) = 0;
    virtual unsigned sample_rate() = 0;
    virtual unsigned number_of_samples() = 0;

    float length() { return number_of_samples() / (float)sample_rate(); }
};
typedef boost::shared_ptr<class Source> pSource;


} // namespace Signal

#endif // SIGNALSOURCE_H
