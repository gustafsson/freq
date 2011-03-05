#ifndef SINKSOURCECHANNELS_H
#define SINKSOURCECHANNELS_H

#include "sinksource.h"

namespace Signal {

/**
  See SinkSource
  */
class SinkSourceChannels : public Sink
{
public:
    SinkSourceChannels();

    void put( pBuffer b );

    void putExpectedSamples( pBuffer b );

    virtual Intervals invalid_samples() { return invalid_samples_all_channels(); }
    Intervals invalid_samples_current_channel();
    Intervals invalid_samples_all_channels();
    virtual void invalidate_samples(const Intervals& I);
    void invalidate_and_forget_samples(const Intervals& I);

    void clear();
    virtual pBuffer read( const Interval& I );
    pBuffer readAllChannelsFixedLength( const Interval& I );
    virtual float sample_rate();
    virtual long unsigned number_of_samples();

    virtual unsigned num_channels();
    virtual void set_channel(unsigned c);
    virtual unsigned get_channel();

    SinkSource& channel(unsigned c);

    void setNumChannels(unsigned);
    pBuffer first_buffer();
    bool empty();
    Intervals samplesDesc() { return samplesDesc_current_channel(); }
    Intervals samplesDesc_current_channel();
    Intervals samplesDesc_all_channels();

private:
    unsigned current_channel_;
    std::vector<SinkSource> sinksources_;
};

} // namespace Signal

#endif // SINKSOURCECHANNELS_H
