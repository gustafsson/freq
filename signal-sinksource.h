#ifndef SIGNALSINKSOURCE_H
#define SIGNALSINKSOURCE_H

#include "signal-sink.h"
#include "signal-source.h"
#include "signal-samplesintervaldescriptor.h"
#include <vector>
#include <QMutex>

namespace Signal {

class SinkSource: public Sink, public Source
{
public:
    SinkSource();

    virtual void put( pBuffer );
    virtual void reset();
    virtual void add_expected_samples( SamplesIntervalDescriptor s );


    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );
    /**
      sample rate is defined as (unsigned)-1 if _cache is empty.
      */
    virtual unsigned sample_rate();
    virtual unsigned number_of_samples();

    pBuffer first_buffer();
    bool empty();
    unsigned size();

    SamplesIntervalDescriptor samplesDesc();

private:
    QMutex _mutex;
    std::vector<pBuffer> _cache;
};

typedef boost::shared_ptr<Sink> pSink;

} // namespace Signal

#endif // SIGNALSINKSOURCE_H
