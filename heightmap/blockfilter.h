#ifndef HEIGHTMAPBLOCKFILTER_H
#define HEIGHTMAPBLOCKFILTER_H

#include "tfr/cwtfilter.h"
#include "tfr/stftfilter.h"
#include "tfr/cepstrumfilter.h"
#include "tfr/drawnwaveformfilter.h"
#include "heightmap/block.h"
#include "heightmap/amplitudeaxis.h"
#include "heightmap/collection.h"

#include <iostream>
#include <vector>

namespace Heightmap
{

class Renderer;


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
                                     bool full_resolution, ComplexInfo complex_info, float normalization_factor, bool enable_subtexel_aggregation );

    unsigned smallestOk(const Signal::Interval& I);

    Collection* _collection;
};


template<typename FilterKind>
class BlockFilterImpl: public FilterKind, public BlockFilter
{
public:
    BlockFilterImpl( Collection* collection )
        :
        BlockFilter(collection),
        largestApplied(0)
    {
    }


    BlockFilterImpl( std::vector<boost::shared_ptr<Collection> >* collections )
        :
        BlockFilter((*collections)[0].get()),
        _collections(collections),
        largestApplied(0)
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

        Signal::Interval I = pchunk.inverse->getInterval();
        largestApplied = std::max( largestApplied, (unsigned)I.count() );
    }


    /// @overload Signal::Operation::affected_samples()
    virtual Signal::Intervals affected_samples()
    {
        return Signal::Intervals::Intervals();
    }


    virtual Signal::pBuffer read(const Signal::Interval& J)
    {
        Signal::Interval I = J;

        // loop, because smallestOk depend on I
        for (
                Signal::Interval K(0,0);
                K != I;
                I = coveredInterval(I))
        {
            K = I;
        }

        return FilterKind::read( I );
    }


    virtual unsigned next_good_size( unsigned current_valid_samples_per_chunk )
    {
        unsigned smallest_ok = smallestOk(Signal::Interval(0,0));
        unsigned requiredSize = std::min(largestApplied, smallest_ok);
        return std::max(requiredSize, FilterKind::next_good_size( current_valid_samples_per_chunk ) );
    }


    virtual unsigned prev_good_size( unsigned current_valid_samples_per_chunk )
    {
        unsigned smallest_ok = smallestOk(Signal::Interval(0,0));
        unsigned requiredSize = std::min(largestApplied, smallest_ok);
        return std::max(requiredSize, FilterKind::prev_good_size( current_valid_samples_per_chunk ) );
    }


    virtual void invalidate_samples(const Signal::Intervals& I)
    {
        largestApplied = 0;
        FilterKind::invalidate_samples( I );
    }


    Signal::Interval coveredInterval(const Signal::Interval& J)
    {
        unsigned smallest_ok = smallestOk(J);
        if (largestApplied < smallest_ok)
        {
            if (0 != J.first)
                undersampled |= J;
        }
        else if (undersampled)
        {
            // don't reset largestApplied, call FilterKind::invalidate_samples directly
            FilterKind::invalidate_samples(undersampled);
            undersampled.clear();
        }

        unsigned requiredSize = std::min(largestApplied, smallest_ok);
        if (requiredSize <= J.count())
            return J;

        // grow in both directions
        Signal::Interval I = Signal::Intervals(J).enlarge( (requiredSize - J.count())/2 ).spannedInterval();
        I.last = I.first + requiredSize;

        if (largestApplied < smallest_ok)
        {
            undersampled |= I;
        }

        return I;
    }

protected:
    std::vector<boost::shared_ptr<Collection> >* _collections;

private:
    unsigned largestApplied;
    Signal::Intervals undersampled;
};


class CwtToBlock: public BlockFilterImpl<Tfr::CwtFilter>
{
public:
    CwtToBlock( std::vector<boost::shared_ptr<Collection> >* collections, Renderer* renderer );

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

private:
    Renderer* renderer;
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
    CepstrumToBlock( std::vector<boost::shared_ptr<Collection> >* collections );

    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
    virtual bool stubWithStft() { return false; }
};


class DrawnWaveformToBlock: public BlockFilterImpl<Tfr::DrawnWaveformFilter>
{
public:
    DrawnWaveformToBlock( std::vector<boost::shared_ptr<Collection> >* collections );

    // @overloads Tfr::DrawnWaveformFilter::computeChunk
    virtual Tfr::ChunkAndInverse computeChunk( const Signal::Interval& I );

    virtual void mergeChunk( pBlock block, Tfr::Chunk& chunk, Block::pData outData );
    virtual bool stubWithStft() { return false; }
    virtual bool createFromOthers() { return false; }
};

} // namespace Heightmap
#endif // HEIGHTMAPBLOCKFILTER_H
