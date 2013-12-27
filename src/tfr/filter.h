#ifndef TFRFILTER_H
#define TFRFILTER_H

#include "signal/intervals.h"
#include "signal/operation.h"
#include "deprecated.h"

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
     * The input buffer used to create 'chunk' with the transform 't'.
     */
    Signal::pMonoBuffer input;


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
    /**
     * @brief The ChunkFilterNoInverse class describes that the inverse shall never
     * be computed from the transformed data in 'ChunkFilter::operator ()'.
     *
     * Inherit from this class as well as from ChunkFilter.
     */
    class NoInverseTag
    {
    public:
        virtual ~NoInverseTag() {}
    };


    virtual ~ChunkFilter() {}

    /**
      Apply the filter to a computed Tfr::Chunk. Return true if it makes sense
      to compute the inverse afterwards.
      */
    virtual void operator()( ChunkAndInverse& chunk ) = 0;

    /**
      Set the number of channels that will get this filter applied.
      May be ignored by the filter if it doesn't matter.
      */
    virtual void set_number_of_channels( unsigned ) {}
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
    virtual Signal::pBuffer process(Signal::pBuffer);
    virtual Signal::Interval requiredInterval( const Signal::Interval& I );
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


    virtual void operator()( ChunkAndInverse& chunk );
    virtual void applyFilter( ChunkAndInverse& chunk );
    virtual void operator()( Chunk& chunk ) = 0;

protected:
    Filter(Filter&);

    /**
     * @brief requiredInterval returns the interval that is required to compute
     * a valid chunk representing interval I.
     * @param I
     * @param t transform() is not invariant, use this instance instead.
     */
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Tfr::pTransform t ) = 0;


private:
    QMutex _transform_mutex;
    /**
      The Tfr::Transform used for computing chunks and inverse Buffers.
      */
    Tfr::pTransform _transform;
};


class TransformKernel: public Signal::Operation
{
public:
    virtual ~TransformKernel() {}

    TransformKernel(Tfr::pTransform t, pChunkFilter chunk_filter);

    virtual Signal::pBuffer process(Signal::pBuffer b);

    Tfr::pTransform transform();
    pChunkFilter chunk_filter();

private:
    Tfr::pTransform transform_;
    pChunkFilter chunk_filter_;
};


class FilterKernelDesc: public VolatilePtr<FilterKernelDesc>
{
public:
    virtual ~FilterKernelDesc() {}

    virtual pChunkFilter createChunkFilter(Signal::ComputingEngine* engine=0) const = 0;
};


class FilterDesc: public Signal::OperationDesc
{
public:
    FilterDesc(Tfr::pTransformDesc, FilterKernelDesc::Ptr);
    virtual ~FilterDesc() {}

    virtual OperationDesc::Ptr copy() const;
    virtual Signal::Operation::Ptr createOperation(Signal::ComputingEngine* engine=0) const;
    virtual Signal::Interval requiredInterval(const Signal::Interval&, Signal::Interval*) const;
    virtual Signal::Interval affectedInterval(const Signal::Interval&) const;
    virtual QString toString() const;
    virtual bool operator==(const Signal::OperationDesc&d) const;

    Tfr::pTransformDesc transformDesc() const;
    virtual void transformDesc(Tfr::pTransformDesc d) { transform_desc_ = d; }
protected:
    Tfr::pTransformDesc transform_desc_;
    FilterKernelDesc::Ptr chunk_filter_;
};

} // namespace Tfr

#endif // TFRFILTER_H
