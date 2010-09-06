#ifndef HEIGHTMAPBLOCKFILTER_H
#define HEIGHTMAPBLOCKFILTER_H

#include "tfr/cwtfilter.h"
#include "tfr/stftfilter.h"
#include "heightmap/collection.h"
#include <iostream>
namespace Heightmap
{

class BlockFilter
{
public:
    BlockFilter( Collection* collection ) :  _collection (collection) {}
    void mergeChunk( Tfr::Chunk& chunk );

protected:
    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData ) = 0;

    Collection* _collection;
};


class CwtToBlock: public Tfr::CwtFilter, public BlockFilter
{
public:
    CwtToBlock( Collection* collection );

    /**
      Tells the "chunk-to-block" what information to extract from the complex
      time-frequency-representation. Such as phase, amplitude or weighted
      amplitude. The weighted ampltidue mode is default for the morlet
      transform to accommodate for low frequencies being smoothed out and
      appear low in amplitude even though they contain frequencies of high
      amplitude.
      */
    ComplexInfo complex_info;

    virtual void operator()( Tfr::Chunk& chunk ) { BlockFilter::mergeChunk(chunk);}
    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
};


class StftToBlock: public Tfr::StftFilter, public BlockFilter
{
public:
    StftToBlock( Collection* collection ) :  BlockFilter(collection) {}

    virtual void operator()( Tfr::Chunk& chunk ) { BlockFilter::mergeChunk(chunk);}
    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
};

} // namespace Heightmap
#endif // HEIGHTMAPBLOCKFILTER_H
