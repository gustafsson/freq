#ifndef SIGNALSINK_H
#define SIGNALSINK_H

#include "signal-source.h"
#include "signal-samplesintervaldescriptor.h"

namespace Signal {

class Sink
{
public:
    virtual ~Sink() {}

    /**
      'put' is the main operation that is done to a sink.

      As an optional parameter a caller may supply a source from which the buffer
      was extracted. the sink may use this information as it may contain valuable
      caches. But the sink is required to perform the same result if only the
      buffer is supplied.
      */
    virtual void put( pBuffer b, pSource=pSource() ) = 0;

    /**
      For some sinks it makes sense to reset, for some it doesn't.
      */
    virtual void reset() {}

    /**
      If this Sink has recieved all expected_samples and is finished with its work, the caller
      may remove this Sink.
      */
    virtual bool finished() { return false; }

    // TODO virtual bool isUnderfed

    /**
      By telling the sink how much data the sink can expect to recieve it is possible
      for the sink to perform some optimizations (such as buffering input before starting
      playing a sound).
      */
    virtual SamplesIntervalDescriptor expected_samples() { return _expected_samples; }
    virtual void add_expected_samples( SamplesIntervalDescriptor s ) { _expected_samples |= s; }

protected:
    /**
      @see expected_samples
      */
    SamplesIntervalDescriptor _expected_samples;
};
typedef boost::shared_ptr<Sink> pSink;

} // namespace Signal

#endif // SIGNALSINK_H
