#ifndef SIGNALFILTEROPERATION_H
#define SIGNALFILTEROPERATION_H

#include "tfr-filter.h"
#include "tfr-cwt.h"

namespace Signal {

class FilterOperation : public Signal::Operation
{
public:
    FilterOperation(Tfr::pFilter filter);

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );

    /**
      Get previous Tfr::Chunk. Used by heightmap rendering.
      */
    Tfr::pChunk previous_chunk() const { return _previous_chunk; }

    /**
      Get/set the Tfr::Filter for this operation.
      */
    Tfr::pFilter    filter() const { return _filter; }
    void            filter( Tfr::pFilter f ) { _filter = f; }

private:

    Tfr::pFilter _filter;
    Tfr::Cwt _cwt;

    Tfr::pChunk _previous_chunk;
};

} // namespace Signal

#endif // SIGNALFILTEROPERATION_H
