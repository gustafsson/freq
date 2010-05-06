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

    // TODO don't keep _previous_chunk unless requested
    /**
      Get previous Tfr::Chunk. Used by heightmap rendering.
      */
    Tfr::pChunk previous_chunk() const { return _previous_chunk; }

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
};

} // namespace Signal

#endif // SIGNALFILTEROPERATION_H
