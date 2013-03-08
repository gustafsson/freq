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
    BlockFilter( Heightmap::TfrMap::Ptr tfr_map );

    virtual bool applyFilter( Tfr::ChunkAndInverse& pchunk);
    unsigned smallestOk(const Signal::Interval& I);
    virtual void mergeChunk( pBlock block, const Tfr::ChunkAndInverse& chunk, Block::pData outData ) = 0;
    virtual bool createFromOthers() { return true; }

protected:
    virtual void mergeColumnMajorChunk( pBlock block, const Tfr::ChunkAndInverse& chunk, Block::pData outData, float normalization_factor );
    virtual void mergeRowMajorChunk( pBlock block, const Tfr::ChunkAndInverse& chunk, Block::pData outData,
                                     bool full_resolution, ComplexInfo complex_info, float normalization_factor, bool enable_subtexel_aggregation );

    Heightmap::TfrMap::Ptr tfr_map_;
};


template<typename FilterKind>
class BlockFilterImpl: public FilterKind, public BlockFilter
{
public:
    BlockFilterImpl( Heightmap::TfrMap::Ptr tfr_map )
        :
        BlockFilter(tfr_map),
        largestApplied(0)
    {
    }


    virtual bool operator()( Tfr::Chunk& )
    {
        return false;
    }


    /// @overload Signal::Operation::affecting_source(const Signal::Interval&)
    Signal::DeprecatedOperation* affecting_source( const Signal::Interval& I)
    {
        Signal::Intervals invalid;

        TfrMap::Collections C = read1(tfr_map_)->collections();
        for (unsigned i=0; i<C.size (); ++i)
            invalid |= write1(C[i])->invalid_samples();

        if (invalid & I)
            return this;

        return FilterKind::source()->affecting_source( I );
    }


    /**
        To prevent anyone from optimizing away a read because it's known to
        result in zeros. BlockFilter wants to be run anyway, even with zeros.
        */
    Signal::Intervals zeroed_samples_recursive() { return Signal::Intervals(); }


    bool applyFilter( Tfr::ChunkAndInverse& pchunk )
    {
        bool r = BlockFilter::applyFilter( pchunk );

        Signal::Interval I = pchunk.inverse->getInterval();
        largestApplied = std::max( largestApplied, (unsigned)I.count() );

        return r;
    }


    /// @overload Signal::Operation::affected_samples()
    virtual Signal::Intervals affected_samples()
    {
        return Signal::Intervals();
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
        unsigned smallest_ok = BlockFilter::smallestOk(Signal::Interval(0,0));
        unsigned requiredSize = std::min(largestApplied, smallest_ok);
        return std::max(requiredSize, FilterKind::next_good_size( current_valid_samples_per_chunk ) );
    }


    virtual unsigned prev_good_size( unsigned current_valid_samples_per_chunk )
    {
        unsigned smallest_ok = BlockFilter::smallestOk(Signal::Interval(0,0));
        unsigned requiredSize = std::min(largestApplied, smallest_ok);
        return std::max(requiredSize, FilterKind::prev_good_size( current_valid_samples_per_chunk ) );
    }


    virtual void invalidate_samples(const Signal::Intervals& I)
    {
        if ((FilterKind::getInterval() - I).empty())
            largestApplied = 0;

        FilterKind::invalidate_samples( I );
    }


    Signal::Interval coveredInterval(const Signal::Interval& J)
    {
        unsigned smallest_ok = BlockFilter::smallestOk(J);
        if (largestApplied < smallest_ok)
        {
            if (!(disregardAtZero() && 0 == J.first))
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
        if (disregardAtZero() && 0==I.first && 0!=J.first)
            I.first = J.first;

        I.last = I.first + requiredSize;


        if (largestApplied < smallest_ok)
        {
            undersampled |= I;
        }

        return I;
    }

protected:
    virtual bool disregardAtZero() { return false; }


private:
    unsigned largestApplied;
    Signal::Intervals undersampled;
};


class CwtToBlock: public BlockFilterImpl<Tfr::CwtFilter>
{
public:
    CwtToBlock( Heightmap::TfrMap::Ptr tfr_map_, Renderer* renderer );

    /**
      Tells the "chunk-to-block" what information to extract from the complex
      time-frequency-representation. Such as phase, amplitude or weighted
      amplitude. The weighted ampltidue mode is default for the morlet
      transform to accommodate for low frequencies being smoothed out and
      appear low in amplitude even though they contain frequencies of high
      amplitude.
      */
    ComplexInfo complex_info;

    virtual void mergeChunk( pBlock block, const Tfr::ChunkAndInverse& chunk, Block::pData outData );
    void mergeChunkpart( pBlock block, const Tfr::ChunkAndInverse& chunk, Block::pData outData );
    virtual bool disregardAtZero() { return true; }

private:
    Renderer* renderer;
};


class StftToBlock: public BlockFilterImpl<Tfr::StftFilter>
{
public:
    StftToBlock( Heightmap::TfrMap::Ptr tfr_map_ );

    Tfr::pChunkFilter freqNormalization;

    bool applyFilter( Tfr::ChunkAndInverse& pchunk )
    {
        // TODO use a chain of commands instead to be processed by the worker thread
        Tfr::pChunkFilter f = freqNormalization;
        if (f)
            f->applyFilter(pchunk);

        if (f != freqNormalization)
            return false;

        return BlockFilterImpl<Tfr::StftFilter>::applyFilter( pchunk );
    }

    virtual void mergeChunk( pBlock block, const Tfr::ChunkAndInverse& chunk, Block::pData outData );
};


class CepstrumToBlock: public BlockFilterImpl<Tfr::CepstrumFilter>
{
public:
    CepstrumToBlock( Heightmap::TfrMap::Ptr tfr_map_ );

    virtual void mergeChunk( pBlock block, const Tfr::ChunkAndInverse& chunk, Block::pData outData );
};


class DrawnWaveformToBlock: public BlockFilterImpl<Tfr::DrawnWaveformFilter>
{
public:
    DrawnWaveformToBlock( Heightmap::TfrMap::Ptr tfr_map_ );

    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Tfr::pTransform t );

    virtual void mergeChunk( pBlock block, const Tfr::ChunkAndInverse& chunk, Block::pData outData );
    virtual bool createFromOthers() { return false; }
};


} // namespace Heightmap
#endif // HEIGHTMAPBLOCKFILTER_H
