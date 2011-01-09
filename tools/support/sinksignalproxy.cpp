#include "sinksignalproxy.h"

#include <QMetaType>

namespace Tools {
    namespace Support {

SinkSignalProxy::
        SinkSignalProxy()
{
    qRegisterMetaType<Signal::Intervals>("Signal::Intervals");
}


void SinkSignalProxy::
        put(Signal::pBuffer b)
{
    emit recievedBuffer( b.get() );
}


void SinkSignalProxy::
        invalidate_samples(const Signal::Intervals& I)
{
    emit recievedInvalidSamples( I );
}


    } // namespace Support
} // namespace Tools
