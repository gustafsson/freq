#ifndef SIGNALPOSTSINK_H
#define SIGNALPOSTSINK_H

#include "signal-sink.h"
#include "tfr-inversecwt.h"
#include <vector>

namespace Signal {

class PostSink: public Sink
{
public:
    virtual void put( pBuffer b ) { put(b, pSource()); }
    virtual void put( pBuffer, pSource );

    virtual void reset();
    virtual bool finished();

    virtual SamplesIntervalDescriptor expected_samples();
    virtual void add_expected_samples( SamplesIntervalDescriptor s );

    Tfr::InverseCwt inverse_cwt;
    std::vector<pSink> sinks;
};

} // namespace Signal

#endif // SIGNALPOSTSINK_H
