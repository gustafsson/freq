#ifndef SIGNALPOSTSINK_H
#define SIGNALPOSTSINK_H

#include "tfr-chunksink.h"
#include "tfr-inversecwt.h"
#include <vector>
#include <QMutex>

namespace Signal {

class PostSink: public Tfr::ChunkSink
{
public:
    virtual void put( pBuffer, pSource );

    virtual void reset();
    virtual bool isFinished();
    virtual void onFinished();

    virtual SamplesIntervalDescriptor expected_samples();
    virtual void add_expected_samples( const SamplesIntervalDescriptor& s );

    std::vector<pSink>  sinks();
    void                sinks(std::vector<pSink> v);
    Tfr::pFilter        filter();
    void                filter(Tfr::pFilter, pSource s);
private:
    Tfr::InverseCwt _inverse_cwt;
    QMutex _sinks_lock;
    std::vector<pSink> _sinks;
};

} // namespace Signal

#endif // SIGNALPOSTSINK_H
