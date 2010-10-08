#ifndef SINKSIGNALPROXY_H
#define SINKSIGNALPROXY_H

#include <QObject>
#include "signal/sink.h"

namespace Tools {
    namespace Support {

        class SinkSignalProxy: public QObject, public Signal::Sink
        {
            Q_OBJECT
        signals:
            void recievedBuffer( Signal::pBuffer );
            void recievedInvalidSamples( const Signal::Intervals& I );

        private:
            virtual bool isFinished() { return false; }
            virtual void put(Signal::pBuffer);
            virtual void invalidate_samples(const Signal::Intervals& I);
        };

    } // namespace Support
} // namespace Tools
#endif // SINKSIGNALPROXY_H
