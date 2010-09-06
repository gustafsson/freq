#ifndef SIGNALSINKSOURCE_H
#define SIGNALSINKSOURCE_H

#include "signal/sink.h"
#include "signal/source.h"
#include "signal/intervals.h"
#include <vector>
#include <QMutex>

namespace Signal {

class SinkSource: public Operation
{
public:
    enum AcceptStrategy {
        AcceptStrategy_ACCEPT_ALL,
        AcceptStrategy_ACCEPT_EXPECTED_ONLY
    };

    SinkSource( AcceptStrategy a );

    void put( pBuffer b );
    void reset();

    virtual pBuffer read( const Interval& I );
    /**
      sample rate is defined as (unsigned)-1 if _cache is empty.
      */
    virtual unsigned sample_rate();
    virtual long unsigned number_of_samples();

    pBuffer first_buffer();
    bool empty();
    unsigned size();

    Intervals samplesDesc();
    virtual void add_expected_samples( const Intervals& s ) { _invalid_samples |= s; }

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
