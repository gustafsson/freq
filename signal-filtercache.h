#ifndef SIGNALFILTERCACHE_H
#define SIGNALFILTERCACHE_H

#include "signal-operation.h"
#include "signal-sinksource.h"

namespace Signal {

/**
  FilterCache is a dummy operation if source is not a FilterOperation.

  If source is a FilterOperation Cache only read from FilterOperation if the
  requested samples have not been previously requested or if the
  SamplesIntervalDescriptors of the filter have changed.
  */


    /**
      TODO scrap above. Instead:
      FilterCache should work this way:
      First read from a pSource, if it is the same as last time it was read
      (save all previous reads) then skip a pSource and instead return the
      previous result. Unless no previous result exist in which case the middle
      pSource gets to do whatever it needs to do.

      Altered filters might for instance invalidate previous results.

      So, a FilterCache reads from source_A first. If source_A returns the same
      as last time and FilterCache has a cached result. Return the cached result.
      Otherwise read from source_B. source_B is set up so that it reads from
      source_A. source_A is of type CachedSource and source_B is of type
      FilterOperation.
      FilterCache saves the result from source_B. The next time FilterCache is
      called the previous result can be returned immediately. Operation::_invalid_samples
      is all that is needed from FilterCache. Thus FilterCache is not explicitly
      dependant on Filter and should have another name.
      */
class FilterCache: public Source
{
public:
    FilterCache( pSource source );

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );

private:
    SinkSource _data;
};


} // namespace Signal

#endif // SIGNALFILTERCACHE_H
