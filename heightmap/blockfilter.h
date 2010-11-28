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
    BlockFilter( Collection* collection );

    /// @overload Tfr::Filter::operator ()(Tfr::Chunk&)
    virtual void operator()( Tfr::Chunk& chunk );

    /// @overload Signal::Operation::affected_samples()
    virtual Signal::Intervals affected_samples();


protected:
    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData ) = 0;
    virtual void computeSlope( Tfr::pChunk chunk );

    Collection* _collection;
};

template<typename FilterKind>
class BlockFilterImpl: public FilterKind, public BlockFilter
{
public:
	BlockFilterImpl( Collection* collection ):BlockFilter(collection) {}

    /// @overload Signal::Operation::fetch_invalid_samples()
	Signal::Intervals fetch_invalid_samples()
	{
		_invalid_samples = _collection->invalid_samples();

		return Tfr::Filter::fetch_invalid_samples();
	}

	virtual void operator()( Tfr::Chunk& chunk ) { BlockFilter::operator()(chunk); }

	/// @overload Signal::Operation::affecting_source(const Signal::Interval&)
	Signal::Operation* affecting_source( const Signal::Interval& ) { return this; }
};

class CwtToBlock: public BlockFilterImpl<Tfr::CwtFilter>
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

    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
    virtual void applyFilter( Tfr::pChunk pchunk );
};


class StftToBlock: public BlockFilterImpl<Tfr::StftFilter>
{
public:
    StftToBlock( Collection* collection ) :  BlockFilterImpl(collection) { _try_shortcuts = false; }

    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
    virtual void applyFilter( Tfr::pChunk pchunk );
};

} // namespace Heightmap
#endif // HEIGHTMAPBLOCKFILTER_H
