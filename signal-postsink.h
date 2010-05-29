#ifndef SIGNALPOSTSINK_H
#define SIGNALPOSTSINK_H

#include "tfr-chunksink.h"
#include "tfr-inversecwt.h"
#include <vector>

namespace Signal {

class PostSink: public Tfr::ChunkSink
{
public:
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
