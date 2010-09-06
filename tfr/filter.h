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
      To simplify logic within Filters they can be put inside an Operation
      group and get their sources set explicitly.
      */
    Filter();
    Filter( Signal::pSource source );

    virtual ~Filter() {}

    virtual Signal::pBuffer read( const Signal::Interval& I );

    virtual void operator()( Chunk& ) = 0;

    virtual Tfr::pTransform transform() const = 0;
    virtual void transform( Tfr::pTransform m ) = 0;


    /**
      These samples are definitely set to 0 by the filter. As default non are
      known to always be set to zero.
      */
    virtual Signal::Intervals ZeroedSamples() const { return Signal::Intervals(); }

    /**
      These samples are needed and possibly affected by the filter.
      ZeroedSamples is assumed to be a subset of AffectedSamples. As default all
      samples are possibly affected by the filter.
      */
    virtual Signal::Intervals AffectedSamples() { return Signal::Intervals::Intervals_ALL; }

    /**
      TODO Define how/when enabled should be used. Should all sources (or all
      Operationshave) an enabled property?
      */
    bool enabled;

protected:
    /**
      Meant to be used between Filters of the same kind to avoid transforming
      back and forth multiple times.
      */
    virtual pChunk readChunk( const Signal::Interval& I ) = 0;

    Tfr::pTransform _transform;
};


} // namespace Tfr

#endif // TFRFILTER_H
