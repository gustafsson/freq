#ifndef TFRCHUNKSINK_H
#define TFRCHUNKSINK_H

#include "signal/sink.h"

#include "filter.h"
#include "transform.h"

namespace Tfr
{

/**
  Virtual class as 'put' is not implemented.

  A class that inherits ChunkSink can use getChunk to convert buffers into chunks. Which tfr transform that will be used is set by get_chunk_transform.
  get_chunk_transform defaults to null which makes it invalid to call getChunk without first setting some transform.
  */
//class ChunkSink: public Signal::Sink
//{
//public:
//    virtual pFilter     chunk_filter()    { return _filter; }
//    virtual void        chunk_filter( pFilter filter )    { _filter = filter; }
//    virtual pTransform  chunk_transform()    { return _transform; }
//    virtual void        chunk_transform( pTransform transform )     {_transform = transform; }

//protected:
//    pFilter     _filter;
//    pTransform  _transform;

//    pChunk getChunk( Signal::pBuffer , Signal::pSource );
//    pChunk getChunkCwt( Signal::pBuffer , Signal::pSource );
//    pChunk getChunkStft( Signal::pBuffer , Signal::pSource );

//    /**
//      If chunks are clamped (cleaned from redundant data) the inverse will produce incorrect results.
//      */
//    static pChunk cleanChunk( pChunk );
//};

} // namespace Tfr

#endif // TFRCHUNKSINK_H
