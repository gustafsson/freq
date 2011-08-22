#ifndef HEIGHTMAPBLOCKFILTER_H
#define HEIGHTMAPBLOCKFILTER_H

#include "tfr/cwtfilter.h"
#include "tfr/stftfilter.h"
#include "tfr/cepstrumfilter.h"
#include "heightmap/collection.h"
#include <iostream>

namespace Heightmap
{


class BlockFilter
{
public:
    BlockFilter( Collection* collection );

    /// @overload Tfr::Filter::operator ()(Tfr::Chunk&)
    //virtual void operator()( Tfr::Chunk& chunk );

    /// @overload Tfr::Filter::applyFilter(Tfr::pChunk)
    virtual void applyFilter(Tfr::pChunk pchunk );

protected:
    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData ) = 0;

    virtual void mergeColumnMajorChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
    virtual void mergeRowMajorChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData,
                                     bool full_resolution, ComplexInfo complex_info );

    Collection* _collection;
};


template<typename FilterKind>
class BlockFilterImpl: public FilterKind, public BlockFilter
{
public:
    BlockFilterImpl( Collection* collection )
        :
        BlockFilter(collection)
    {
    }


    BlockFilterImpl( std::vector<boost::shared_ptr<Collection> >* collections )
        :
        BlockFilter((*collections)[0].get()),
        _collections(collections)
    {
    }


    virtual void operator()( Tfr::Chunk& )
    {
        BOOST_ASSERT( false );
    }


    /// @overload Signal::Operation::affecting_source(const Signal::Interval&)
    Signal::Operation* affecting_source( const Signal::Interval& I)
    {
        if (_collection->invalid_samples() & I)
            return this;

        return FilterKind::source()->affecting_source( I );
    }


    /**
        To prevent anyone from optimizing away a read because it's known to
        result in zeros. BlockFilter wants to be run anyway, even with zeros.
        */
    Signal::Intervals zeroed_samples_recursive() { return Signal::Intervals(); }

    virtual void set_channel(unsigned c)
    {
        FilterKind::set_channel(c);

        _collection = (*_collections)[c].get();
    }

    void applyFilter( Tfr::pChunk pchunk )
    {
        BlockFilter::applyFilter( pchunk );
    }


    /// @overload Signal::Operation::affected_samples()
    virtual Signal::Intervals affected_samples()
    {
        return Signal::Intervals::Intervals();
    }

protected:
    std::vector<boost::shared_ptr<Collection> >* _collections;
};


class CwtToBlock: public BlockFilterImpl<Tfr::CwtFilter>
{
public:
    CwtToBlock( Collection* collection );
    CwtToBlock( std::vector<boost::shared_ptr<Collection> >* collections );

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
    void mergeChunkpart( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
};


class StftToBlock: public BlockFilterImpl<Tfr::StftFilter>
{
public:
    StftToBlock( Collection* collection );
    StftToBlock( std::vector<boost::shared_ptr<Collection> >* collections );

    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
};


class CepstrumToBlock: public BlockFilterImpl<Tfr::CepstrumFilter>
{
public:
    CepstrumToBlock( Collection* collection );
    CepstrumToBlock( std::vector<boost::shared_ptr<Collection> >* collections );

    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
};

} // namespace Heightmap
#endif // HEIGHTMAPBLOCKFILTER_H
