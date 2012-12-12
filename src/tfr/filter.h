#ifndef TFRFILTER_H
#define TFRFILTER_H

#include "signal/intervals.h"
#include "signal/operation.h"

#include <QMutex>

namespace Tfr {

class Transform;
typedef boost::shared_ptr<Transform> pTransform;
class TransformDesc;
typedef boost::shared_ptr<TransformDesc> pTransformDesc;

class Chunk;
typedef boost::shared_ptr<Chunk> pChunk;

/// @see ChunkAndInverse::inverse
struct ChunkAndInverse
{
    /**
     * The transform used to compute the chunk.
     */
    pTransform t;

    /**
      The Tfr::Chunk as computed by readChunk(), or source()->readChunk()
      if transform() == source()->transform().
      */
    pChunk chunk;


    /**
      The variable 'inverse' _may_ be set by readChunk if
      this->source()->readFixed(chunk->getInterval()) is identical to
      this->transform()->inverse(chunk). In that case the inverse won't be
      computed again.
      */
    Signal::pMonoBuffer inverse;


    /**
     * Which channel the monobuffer comes from.
     */
    int channel;
};


/**
 * @brief The ChunkFilter class
 */
class ChunkFilter
{
public:
    virtual ~ChunkFilter() {}

    /**
      The default implementation of applyFilter is to call operator()( Chunk& )
      @see computeChunk
      */
    virtual bool applyFilter( ChunkAndInverse& chunk );

protected:
    /**
      Apply the filter to a computed Tfr::Chunk. This is the method that should
      be implemented to create new filters. Return true if it makes sense to
      compute the inverse afterwards.
      */
    virtual bool operator()( Chunk& ) = 0;
};
typedef boost::shared_ptr<ChunkFilter> pChunkFilter;


/**
  Virtual base class for filters. To create a new filter, use CwtFilter or
  StftFilter as base class and implement the method 'operator()( Chunk& )'.
  */
class Filter: public Signal::DeprecatedOperation, public ChunkFilter
{
public:
    /**
      TODO verify/implement
      To simplify logic within Filters they can be put inside an Operation
      group and get their sources set explicitly.
      */
    Filter( Signal::pOperation source = Signal::pOperation() );


    /**
      Checks if the requested Signal::Interval would be affected by this filter
      (using Signal::Operation::affected_samples()) and if so calls
      readChunk().

      @remarks If _try_shortcuts is true, readChunk() is always called,
      regardless of Signal::Operation::affected_samples().

      @overload Operation::read(const Signal::Interval&)
      */
    virtual Signal::pBuffer read( const Signal::Interval& I );


    /**
      Filters are applied to chunks that are computed using some transform.
      transform()/transform(pTransform) gets/sets that transform.

      For thread safety it is important to only call transform() once during
      a computation pass. Subsequent calls to transform() might return
      different transforms.
      */
    Tfr::pTransform transform();


    /**
      Set the Tfr::Transform for this operation and call invalidate_samples.
      Will throw throw std::invalid_argument if the type of 'm' is not equal to
      the previous type of transform().
      */
    void transform( Tfr::pTransform m );


    /**
      If _try_shortcuts is true. This method from Operation will be used to
      try to avoid computing any actual transform.
      */
    virtual DeprecatedOperation* affecting_source( const Signal::Interval& I );


    /**
      Returns the next good chunk size for the transform() (or the
      largest if there is no good chunk size larger than
      'current_valid_samples_per_chunk').
      */
    virtual unsigned next_good_size( unsigned current_valid_samples_per_chunk );


    /**
      Returns the previously good chunk size for transform() (or the
      smallest if there is no good chunk size larger than
      'current_valid_samples_per_chunk').
      */
    virtual unsigned prev_good_size( unsigned current_valid_samples_per_chunk );

protected:
    Filter(Filter&);

    /**
     * @brief requiredInterval returns the interval that is required to compute
     * a valid chunk representing interval I.
     * @param I
     * @param t transform() is not invariant use this instance instead.
     */
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Tfr::pTransform t ) = 0;


private:
    QMutex _transform_mutex;
    /**
      The Tfr::Transform used for computing chunks and inverse Buffers.
      */
    Tfr::pTransform _transform;
};


} // namespace Tfr

#endif // TFRFILTER_H
