#ifndef BUFFER_H
#define BUFFER_H

#include "intervals.h"

// gpumisc
#include "datastorage.h"
#include "unsignedf.h"

// boost
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

namespace Signal {


/**
The Signal::Buffer class is _the_ container class for data in the Signal
namespace. A Buffer can contain an entire song as when created by
Signal::Audiofile, or a Buffer can contain sound as fractions of a second
as when created by Signal::MicrophoneRecorder.
*/
class SaweDll Buffer : public boost::noncopyable {
public:
    Buffer(UnsignedF firstSample,
           IntervalType numberOfSamples,
           float sample_rate,
           unsigned numberOfChannels = 1,
           unsigned numberOfSignals = 1);
    /**
      Always creates a single channel buffer.
      */
    Buffer(Signal::Interval subinterval, boost::shared_ptr<Buffer> other, unsigned channel=0);
    ~Buffer();

    DataStorage<float>::Ptr  waveform_data() const;
    IntervalType        number_of_samples() const
    {
        return waveform_data_->size().width;
    }
    void                release_extra_resources();

    UnsignedF       sample_offset;
    float           sample_rate;

    float           start() const;
    float           length() const;
    Interval        getInterval() const;

    unsigned        channels() const;

    /// element-wise overwrite 'this' with data from 'b' where they overlap
    Buffer&         operator|=(const Buffer& b);
    /// element-wise add 'this' with 'b' where they overlap
    Buffer&         operator+=(const Buffer& b);

protected:
    DataStorage<float>::Ptr waveform_data_;
    boost::shared_ptr<Buffer> other_;
    unsigned bitor_channel_;
};
typedef boost::shared_ptr<Buffer> pBuffer;

} // namespace Signal

#endif // BUFFER_H
