#ifndef TFRFILTER_H
#define TFRFILTER_H

#include "signal/operation.h"

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


class TransformKernel: public Signal::Operation
{
public:
    TransformKernel(Tfr::pTransform t, pChunkFilter chunk_filter);

    // Signal::Operation
    Signal::pBuffer process(Signal::pBuffer b);

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
