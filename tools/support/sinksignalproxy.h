#ifndef SINKSIGNALPROXY_H
#define SINKSIGNALPROXY_H

#include <QObject>
#include "signal/sink.h"

namespace Tools {
    namespace Support {

        class SinkSignalProxy: public QObject, public Signal::Sink
        {
            Q_OBJECT
        public:
            SinkSignalProxy();

        signals:
            void recievedBuffer( Signal::Buffer* );
            void recievedInvalidSamples( Signal::Intervals );

        private:
            virtual bool deleteMe() { return false; }
            virtual void put(Signal::pBuffer);
            virtual void invalidate_samples(const Signal::Intervals& I);
        };

    } // namespace Support
} // namespace Tools
#endif // SINKSIGNALPROXY_H
