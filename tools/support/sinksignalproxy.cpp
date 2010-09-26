#include "sinksignalproxy.h"

namespace Tools {
    namespace Support {

void SinkSignalProxy::
        put(Signal::pBuffer b)
{
    emit recievedBuffer( b );
}


void SinkSignalProxy::
        invalidate_samples(const Signal::Intervals& I)
{
    emit recievedInvalidSamples( I );
}


    } // namespace Support
} // namespace Tools
