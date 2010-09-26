#ifndef TFRFILTER_H
#define TFRFILTER_H

#include "signal/intervals.h"
#include "signal/operation.h"
#include "tfr/transform.h"

#include <list>
#include <boost/shared_ptr.hpp>

namespace Tfr {


/**
  Virtual base class for filters. To create a new filter, use CwtFilter or
  StftFilter as base class and implement the method 'operator()( Chunk& )'.
  */
class Filter: public Signal::Operation
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
      */
    virtual Tfr::pTransform transform() const = 0;


    /// @see transform()
    virtual void transform( Tfr::pTransform m ) = 0;


    /**
      These samples are definitely set to 0 by the filter. As default none are
      known to always be set to zero.

      @remarks zeroed_samples is _assumed_ (but never checked) to be a subset
      of Signal::Operation::affected_samples().
      */
    virtual Signal::Intervals zeroed_samples() const { return Signal::Intervals(); }


protected:
    /**
      Apply the filter to a computed Tfr::Chunk. This is the method that should
      be implemented to create new filters.
      */
    virtual void operator()( Chunk& ) = 0;


    /// @see ChunkAndInverse::inverse
    struct ChunkAndInverse
    {
        /**
          The Tfr::Chunk as computed by readChunk(), or source()->readChunk()
          if transform() == source()->transform().
          */
        pChunk chunk;


        /**
          The variable 'inverse' _may_ be set by readChunk if
          this->source()->readFixed(chunk->getInterval()) is identical to
          this->_transform->inverse(chunk). In that case the inverse won't be
          computed again.
          */
        Signal::pBuffer inverse;
    };


    /**
      Meant to be used between Filters of the same kind to avoid transforming
      back and forth multiple times.
      */
    virtual ChunkAndInverse readChunk( const Signal::Interval& I ) = 0;


    /**
      _try_shortcuts is set to false by an implementation if it requires that
      all chunks be computed, even if the filter won't affect any samples when
      the inverse is computed. If _try_shortcuts is false,
      ChunkAndInverse::inverse _may_ contain the original Buffer as it were
      before the chunk was computed.

      _try_shortcuts defaults to true.
      */
    bool _try_shortcuts;


    /**
      The Tfr::Transform used for computing chunks and inverse Buffers.
      */
    Tfr::pTransform _transform;
};


} // namespace Tfr

#endif // TFRFILTER_H
