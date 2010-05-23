#ifndef SAMPLESINTERVALDESCRIPTOR_H
#define SAMPLESINTERVALDESCRIPTOR_H

#include <list>
#include <stdexcept> // TODO remove temp code

namespace Signal {

class SamplesIntervalDescriptor
{
public:
    typedef unsigned SampleType;
    static const SampleType SampleType_MIN;
    static const SampleType SampleType_MAX;
    static const SamplesIntervalDescriptor SamplesIntervalDescriptor_ALL;

    struct Interval {
        /**
            'last' is rather the first sample not within the interval, such that
            the length of the interval can be computed as "last-first".
          */
        SampleType first, last;

        bool operator<(const Interval& r) const;
        bool operator|=(const Interval& r);
        bool operator==(const Interval& r) const;
    };

    SamplesIntervalDescriptor( );
    SamplesIntervalDescriptor( Interval );
    SamplesIntervalDescriptor( float , float )
    {
        throw std::runtime_error("NOOO2");
    }
    SamplesIntervalDescriptor( SampleType first, SampleType last );
    SamplesIntervalDescriptor& operator |= (const SamplesIntervalDescriptor&);
    SamplesIntervalDescriptor& operator |= (const Interval&);
    SamplesIntervalDescriptor& operator -= (const SamplesIntervalDescriptor&);
    SamplesIntervalDescriptor& operator -= (const Interval&);
    SamplesIntervalDescriptor& operator -= (const SampleType&);
    SamplesIntervalDescriptor& operator += (const SampleType&);
    SamplesIntervalDescriptor& operator &= (const SamplesIntervalDescriptor&);
    SamplesIntervalDescriptor& operator &= (const Interval&);
    SamplesIntervalDescriptor& operator *= (const float& scale);

    Interval    getInterval( SampleType dt, SampleType center = 0 ) const;

    bool isEmpty() const { return _intervals.empty(); }
    const std::list<Interval>& intervals() { return _intervals; }
private:
    std::list<Interval> _intervals;
};

} // namespace Signal

#endif // SAMPLESINTERVALDESCRIPTOR_H
