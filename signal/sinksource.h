#ifndef SIGNALSINKSOURCE_H
#define SIGNALSINKSOURCE_H

#include "signal/sink.h"
#include "signal/source.h"
#include "signal/samplesintervaldescriptor.h"
#include <vector>
#include <QMutex>

namespace Signal {

class SinkSource: public Sink, public Source
{
public:
    enum AcceptStrategy {
        AcceptStrategy_ACCEPT_ALL,
        AcceptStrategy_ACCEPT_EXPECTED_ONLY
    };

    SinkSource( AcceptStrategy a );

    void put( pBuffer );
    virtual void put( pBuffer b, pSource ) { put (b); }
    virtual void reset();

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );
    /**
      sample rate is defined as (unsigned)-1 if _cache is empty.
      */
    virtual unsigned sample_rate();
    virtual long unsigned number_of_samples();

    pBuffer first_buffer();
    bool empty();
    unsigned size();

    SamplesIntervalDescriptor samplesDesc();

private:
    QMutex _cache_mutex;
    std::vector<pBuffer> _cache;
    AcceptStrategy _acceptStrategy;

    void selfmerge();
    void merge( pBuffer );
};

typedef boost::shared_ptr<Sink> pSink;

} // namespace Signal

#endif // SIGNALSINKSOURCE_H
