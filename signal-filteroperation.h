#ifndef SIGNALFILTEROPERATION_H
#define SIGNALFILTEROPERATION_H

#include "tfr-filter.h"
#include "tfr-cwt.h"
#include "tfr-inversecwt.h"
#include "signal-operation.h"

namespace Signal {

class FilterOperation : public Signal::Operation
{
public:
    FilterOperation( pSource source, Tfr::pFilter filter);

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );

    /**
      Pick previous Tfr::Chunk. Used by heightmap rendering. Not guaranteed to
      return a chunk, will return null unless polled before each call to 'read'
      (and 'read' concluded that a chunk had to be computed).

      Also returns a chunk only once. Two subsequent calls to pick_... without
      calling 'read' in between will make the second call return null.
      */
    Tfr::pChunk pick_previous_chunk();

    /**
      Get/set the Tfr::Filter for this operation.
      */
    Tfr::pFilter    filter() const { return _filter; }
    void            filter( Tfr::pFilter f ) { _filter = f; }

    Tfr::Cwt cwt;
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
