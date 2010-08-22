#ifndef TFRCHUNKSINK_H
#define TFRCHUNKSINK_H

#include "signal/sink.h"
#include "tfr/chunk.h"
#include "tfr/filter.h"

namespace Tfr {

class ChunkSink: public Signal::Sink
{
public:
	pFilter get_chunk_filter;
protected:
    pChunk getChunk( Signal::pBuffer , Signal::pSource );

    /**
      If chunks are clamped (cleaned from redundant data) the inverse will produce incorrect results.
      */
    static pChunk cleanChunk( pChunk );
};

} // namespace Tfr

#endif // TFRCHUNKSINK_H
