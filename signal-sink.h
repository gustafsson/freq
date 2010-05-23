#ifndef SIGNALSINK_H
#define SIGNALSINK_H

#include "signal-source.h"

namespace Signal {

class Sink
{
public:
    Sink();
    virtual ~Sink() {}

    /**
      'put' is the main operation that is done to a sink.
      */
    virtual void put( pBuffer ) = 0;

    /**
      As an optional parameter a caller may supply a source from which the buffer
      was extracted. the sink may use this information as it may contain valuable
      caches. But the sink is required to perform the same result if only the
      buffer is supplied.
      */
    virtual void put( pBuffer b, pSource ) { put (b); }

    /**
      For some sinks it makes sense to reset, for some it doesn't.
      */
    virtual void reset() {}

    /**
      By telling the sink how much data the sink can expect to recieve it is possible
      for the sink to perform some optimizations (such as buffering input before starting
      playing a sound).
      */
    virtual void expected_samples_left(unsigned);
    virtual unsigned expected_samples_left();

private:
    /**
      @see expected_samples_left
      */
    unsigned _expected_samples_left;
};
typedef boost::shared_ptr<Sink> pSink;

} // namespace Signal

#endif // SIGNALSINK_H
