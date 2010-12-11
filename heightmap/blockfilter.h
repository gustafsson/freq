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

protected:
    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData ) = 0;
    virtual void computeSlope( Tfr::pChunk chunk );

    Collection* _collection;
};

template<typename FilterKind>
class BlockFilterImpl: public FilterKind, public BlockFilter
{
public:
    BlockFilterImpl( Collection* collection ) : BlockFilter(collection)  { }
    BlockFilterImpl( std::vector<boost::shared_ptr<Collection> > collections )
        :
        BlockFilter(collections[0].get()),
        _collections(collections)
    {
    }

    /// @overload Signal::Operation::fetch_invalid_samples()
    Signal::Intervals fetch_invalid_samples()
    {
        if (FilterKind::_invalid_samples)
        {
            foreach ( boost::shared_ptr<Collection> c, _collections)
                _collection->invalidate_samples( FilterKind::_invalid_samples );

            FilterKind::_invalid_samples.clear();
        }

        foreach ( boost::shared_ptr<Collection> c, _collections)
        {
            FilterKind::_invalid_samples |= _collection->invalid_samples();
        }

        return Tfr::Filter::fetch_invalid_samples();
    }


    virtual void operator()( Tfr::Chunk& chunk )
    {
        Signal::FinalSource * fs = dynamic_cast<Signal::FinalSource*>(FilterKind::root());
        BOOST_ASSERT( fs );

        _collection = _collections[fs->get_channel()].get();

        BlockFilter::operator()(chunk);
    }

    /// @overload Signal::Operation::affecting_source(const Signal::Interval&)
    Signal::Operation* affecting_source( const Signal::Interval& ) { return this; }

    void applyFilter( Tfr::pChunk pchunk )
    {
        FilterKind::applyFilter( pchunk );

        computeSlope( pchunk );
    }

    /// @overload Signal::Operation::affected_samples()
    virtual Signal::Intervals affected_samples()
    {
        return Signal::Intervals::Intervals();
    }

protected:
    std::vector<boost::shared_ptr<Collection> > _collections;
};

class CwtToBlock: public BlockFilterImpl<Tfr::CwtFilter>
{
public:
    CwtToBlock( Collection* collection );
    CwtToBlock( std::vector<boost::shared_ptr<Collection> > collections );

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
};


class StftToBlock: public BlockFilterImpl<Tfr::StftFilter>
{
public:
    StftToBlock( Collection* collection ) :  BlockFilterImpl<Tfr::StftFilter>(collection) { _try_shortcuts = false; }
    StftToBlock( std::vector<boost::shared_ptr<Collection> > collections ) :  BlockFilterImpl<Tfr::StftFilter>(collections) { _try_shortcuts = false; }

    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
};

} // namespace Heightmap
#endif // HEIGHTMAPBLOCKFILTER_H
