#ifndef TFRFILTER_H
#define TFRFILTER_H

#include "sawe/selection.h"
#include "signal/intervals.h"
#include "signal/operation.h"
#include "tfr/transform.h"

#include <list>
#include <boost/shared_ptr.hpp>

namespace Tfr {


class Filter: public Signal::Operation
{
public:
    /**
      TDOO verify
      To simplify logic within Filters they can be put inside an Operation
      group and get their sources set explicitly.
      */
    Filter( Signal::pOperation source = Signal::pOperation() );

    virtual ~Filter() {}

    virtual Signal::pBuffer read( const Signal::Interval& I );

    virtual Tfr::pTransform transform() const = 0;
    virtual void transform( Tfr::pTransform m ) = 0;


    /**
      These samples are definitely set to 0 by the filter. As default none are
      known to always be set to zero.

      zeroed_samples is assumed to be a subset of affected_samples.
      */
    virtual Signal::Intervals zeroed_samples() const { return Signal::Intervals(); }


protected:
    virtual void operator()( Chunk& ) = 0;

    /**
      Meant to be used between Filters of the same kind to avoid transforming
      back and forth multiple times.
      */
    virtual pChunk readChunk( const Signal::Interval& I ) = 0;


    Tfr::pTransform _transform;
};


} // namespace Tfr

#endif // TFRFILTER_H
