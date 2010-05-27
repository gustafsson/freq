#ifndef TFRCHUNKSINK_H
#define TFRCHUNKSINK_H

#include "signal-sink.h"
#include "tfr-chunk.h"

namespace Tfr {

class ChunkSink: public Signal::Sink
{
public:
    void    put( Signal::pBuffer b ) { put(b, Signal::pSource()); }
    void    put( Signal::pBuffer , Signal::pSource ) = 0;
protected:
    pChunk getChunk( Signal::pBuffer , Signal::pSource );
    static pChunk cleanChunk( pChunk );
};

} // namespace Tfr

#endif // TFRCHUNKSINK_H
