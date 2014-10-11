#ifndef BUFFER_H
#define BUFFER_H

#include "intervals.h"

// gpumisc
#include "datastorage.h"
#include "unsignedf.h"

// boost
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

// std
#include <vector>

namespace Signal {

typedef DataStorage<float> TimeSeriesData;
typedef TimeSeriesData::ptr pTimeSeriesData;

class SignalDll MonoBuffer : public boost::noncopyable {
public:
    MonoBuffer(Interval I, float sample_rate);
    MonoBuffer(UnsignedF first_sample, pTimeSeriesData ptr, float sample_rate);
    MonoBuffer(UnsignedF first_sample, IntervalType number_of_samples, float sample_rate);
    ~MonoBuffer();

    pTimeSeriesData         waveform_data() const { return time_series_; }
    IntervalType            number_of_samples() const { return time_series_->size().width; }
    void                    release_extra_resources();

    UnsignedF               sample_offset() const { return sample_offset_; }
    float                   sample_rate() const { return sample_rate_; }
    void                    set_sample_rate(float fs) { sample_rate_ = fs; }
    void                    set_sample_offset(UnsignedF offset) { sample_offset_ = offset; }

    double                   start() const;
    float                   length() const;
    Interval                getInterval() const;

    /// element-wise overwrite 'this' with data from 'b' where they overlap
    MonoBuffer&             operator|=(MonoBuffer const& b);
    /// element-wise add 'this' with 'b' where they overlap
    MonoBuffer&             operator+=(MonoBuffer const& b);

    // Used for testing, compares on the CPU
    bool                    operator==(MonoBuffer const& b) const;
    bool                    operator!=(MonoBuffer const& b) const { return !(*this == b); }

private:
    // Not implemented, no copying
    MonoBuffer(const MonoBuffer&);

    pTimeSeriesData time_series_;
    UnsignedF       sample_offset_;
    float           sample_rate_;
};

typedef boost::shared_ptr<MonoBuffer> pMonoBuffer;


/**
The Signal::Buffer class is _the_ container class for data in the Signal
namespace. A Buffer can contain an entire song as when created by
Signal::Audiofile, or a Buffer can contain sound as fractions of a second
as when created by Signal::MicrophoneRecorder.
*/
class SignalDll Buffer : public boost::noncopyable {
public:
    Buffer(Interval I, float sample_rate, int number_of_channels);
    Buffer(UnsignedF first_sample,
           IntervalType number_of_samples,
           float sample_rate,
           int number_of_channels);
    Buffer(Buffer&& b);
    explicit Buffer(pMonoBuffer b);
    Buffer(UnsignedF first_sample, pTimeSeriesData ptr, float sample_rate);
    ~Buffer();

    IntervalType    number_of_samples() const { return getChannel(0)->number_of_samples (); }
    // TODO change type to int
    unsigned        number_of_channels() const { return channels_.size(); }
    void            release_extra_resources();

    UnsignedF       sample_offset() const { return getChannel(0)->sample_offset(); }
    float           sample_rate() const { return getChannel(0)->sample_rate(); }
    void            set_sample_rate(float);
    void            set_sample_offset(UnsignedF offset);

    float           start() const { return getChannel(0)->start(); }
    float           length() const { return getChannel(0)->length(); }
    Interval        getInterval() const { return getChannel(0)->getInterval(); }

    pMonoBuffer     getChannel(int channel) const { return channels_[channel]; }
    pTimeSeriesData mergeChannelData() const;

    /// element-wise overwrite 'this' with data from 'b' where they overlap
    Buffer&         operator|=(const Buffer& b);
    /// element-wise add 'this' with 'b' where they overlap
    Buffer&         operator+=(const Buffer& b);

    // Used for testing
    bool            operator==(const Buffer& b) const;
    bool            operator!=(const Buffer& b) const { return !(*this == b); }

    static void     test();
private:
    std::vector<pMonoBuffer> channels_;
};

typedef boost::shared_ptr<Buffer> pBuffer;

} // namespace Signal

#endif // BUFFER_H
