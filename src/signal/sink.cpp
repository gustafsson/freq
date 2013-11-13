#include "sink.h"

namespace Signal {

    pBuffer Sink::read(const Interval& I) {
        EXCEPTION_ASSERT(source());

        pBuffer b = source()->read(I);

        // Check if read returned the first sample in interval I
        Interval i(I.first, I.first + 1);
        EXCEPTION_ASSERT( (i & b->getInterval()) == i );

        put(b);
        //_invalid_samples -= b->getInterval();
        return b;
    }


    void Sink::put(pBuffer) {
        throw std::logic_error(
            "Neither read nor put seems to have been overridden from Sink in " + vartype(*this) + ".");
    }


    // static
    pBuffer Sink::put(DeprecatedOperation* receiver, pBuffer buffer) {
        pOperation s( new BufferSource(buffer));
        pOperation old = receiver->source();
        receiver->source(s);
        pBuffer r = receiver->read(buffer->getInterval());
        receiver->source(old);
        return r;
    }

}
