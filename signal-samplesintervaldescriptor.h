#ifndef SAMPLESINTERVALDESCRIPTOR_H
#define SAMPLESINTERVALDESCRIPTOR_H

#include <list>

namespace Signal {

class SamplesIntervalDescriptor
{
public:
    typedef unsigned SampleType;

    struct Interval {
        /**
            'last' is rather the first sample not within the interval, such that
            the length of the interval can be computed as "last-first".
          */
        SampleType first, last;

        bool operator<(const Interval& r) const;
        bool operator|=(const Interval& r);
    };

    SamplesIntervalDescriptor( );
    SamplesIntervalDescriptor( Interval );
    SamplesIntervalDescriptor( SampleType first, SampleType last );
    SamplesIntervalDescriptor& operator |= (const SamplesIntervalDescriptor&);
    SamplesIntervalDescriptor& operator |= (const Interval&);
    SamplesIntervalDescriptor& operator -= (const SamplesIntervalDescriptor&);
    SamplesIntervalDescriptor& operator -= (const Interval&);
    SamplesIntervalDescriptor& operator &= (const SamplesIntervalDescriptor&);
    SamplesIntervalDescriptor& operator &= (const Interval&);
    SamplesIntervalDescriptor& operator *= (const float& scale);

    Interval    popInterval( SampleType dt, SampleType center = 0 );

    const std::list<Interval>& intervals() { return _intervals; }
private:
    std::list<Interval> _intervals;
};

} // namespace Signal

#endif // SAMPLESINTERVALDESCRIPTOR_H
