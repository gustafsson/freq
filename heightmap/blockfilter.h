#ifndef HEIGHTMAPBLOCKFILTER_H
#define HEIGHTMAPBLOCKFILTER_H

#include "tfr/cwtfilter.h"
#include "tfr/stftfilter.h"
#include "tfr/cepstrumfilter.h"
#include "tfr/drawnwaveformfilter.h"
#include "heightmap/collection.h"
#include <iostream>

namespace Heightmap
{


class BlockFilter
{
public:
    BlockFilter( Collection* collection );

    virtual void applyFilter(Tfr::ChunkAndInverse& pchunk );
    virtual bool stubWithStft() { return true; }
    virtual bool createFromOthers() { return true; }

protected:
    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData ) = 0;

    virtual void mergeColumnMajorChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData, float normalization_factor );
    virtual void mergeRowMajorChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData,
                                     bool full_resolution, ComplexInfo complex_info, float normalization_factor );

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

    void applyFilter( Tfr::ChunkAndInverse& pchunk )
    {
        BlockFilter::applyFilter( pchunk );

        Signal::Interval I = transform()->validLength(pchunk.inverse);
        pchunk.inverse.reset( new Signal::Buffer(I.first, I.count(), pchunk.inverse->sample_rate) );
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
    virtual bool stubWithStft() { return false; }
};


class CepstrumToBlock: public BlockFilterImpl<Tfr::CepstrumFilter>
{
public:
    CepstrumToBlock( Collection* collection );
    CepstrumToBlock( std::vector<boost::shared_ptr<Collection> >* collections );

    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
    virtual bool stubWithStft() { return false; }
};


class DrawnWaveformToBlock: public BlockFilterImpl<Tfr::DrawnWaveformFilter>
{
public:
    DrawnWaveformToBlock( Collection* collection );
    DrawnWaveformToBlock( std::vector<boost::shared_ptr<Collection> >* collections );

    // @overloads Tfr::DrawnWaveformFilter::computeChunk
    virtual Tfr::ChunkAndInverse computeChunk( const Signal::Interval& I );

    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
    virtual bool stubWithStft() { return false; }
    virtual bool createFromOthers() { return false; }
};

} // namespace Heightmap
#endif // HEIGHTMAPBLOCKFILTER_H
