#include "signal-filtercache.h"

namespace Signal {

FilterCache::
        FilterCache( pSource source )
:   _data(source)
{

}

pBuffer FilterCache::
        read( unsigned firstSample, unsigned numberOfSamples )
{
    // Check filters
    return _data.read( firstSample, numberOfSamples );
}

} // namespace Signal
