#ifndef SIGNALFILTEROPERATION_H
#define SIGNALFILTEROPERATION_H

#include "tfr-filter.h"
#include "tfr-cwt.h"
#include "tfr-inversecwt.h"
#include "signal-operationcache.h"

namespace Signal {

class FilterOperation : public Signal::OperationCache
{
public:
    FilterOperation( pSource source, Tfr::pFilter filter);

    virtual pBuffer readRaw( unsigned firstSample, unsigned numberOfSamples );
    virtual bool cacheMiss(unsigned firstSample, unsigned numberOfSamples);

    /**
      Pick previous Tfr::Chunk. Used by heightmap rendering. Not guaranteed to
      return a chunk, will return null unless polled before each call to 'read'
      (and 'read' concluded that a chunk had to be computed).

      When previous_chunk is polled before calling read, read will not compute
      the inverse of a chunk. The caller may if need compute the inverse by

        pBuffer b = FilterOperation::inverse_cwt(*FilterOperation::previous_chunk())
      */
    Tfr::pChunk previous_chunk();

    /**
      Releases previous chunk prior to calling read.
      */
    void release_previous_chunk();

    /**
      Get the Tfr::Filter for this operation.
      */
    Tfr::pFilter    filter() const { return _filter; }

    /**
      Set the Tfr::Filter for this operation and update _invalid_samples.
      */
    void            filter( Tfr::pFilter f );

    Tfr::InverseCwt inverse_cwt;

    /**
      If source also is a FilterOperation, take out its pFilter and do both
      filters in this FilterOperation. Then remove source by taking source->source
      as source instead.
      */
    void meldFilters();

private:
    Tfr::pFilter _filter;
    Tfr::pChunk _previous_chunk;
    bool _save_previous_chunk;
};

/*class OperationSkip: public Signal::Operation
{
public:
    OperationSkip( pSource source, pSource baseSource, SamplesIntervalDescriptor skip, bool setToZero);

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );

private:
    SamplesIntervalDescriptor _skipped;
};*/

} // namespace Signal

#endif // SIGNALFILTEROPERATION_H
