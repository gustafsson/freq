#ifndef SIGNALSINK_H
#define SIGNALSINK_H

#include "signal-source.h"

namespace Signal {

class Sink
{
public:
    Sink();
    virtual ~Sink() {}

    virtual void reset() = 0;
    virtual void put( pBuffer ) = 0;
    virtual unsigned expected_samples_left();
    virtual void expected_samples_left(unsigned);
private:
    unsigned _expected_samples_left;
};
typedef boost::shared_ptr<Sink> pSink;

} // namespace Signal

#endif // SIGNALSINK_H
