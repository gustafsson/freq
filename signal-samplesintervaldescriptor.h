#ifndef SAMPLESINTERVALDESCRIPTOR_H
#define SAMPLESINTERVALDESCRIPTOR_H

#include <list>
#include <ostream>

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

        bool valid() const;
        bool isConnectedTo(const Interval& r) const;
        bool operator<(const Interval& r) const;
        Interval& operator|=(const Interval& r);
        bool operator==(const Interval& r) const;
    };

    static const SampleType SampleType_MIN;
    static const SampleType SampleType_MAX;
    static const SamplesIntervalDescriptor SamplesIntervalDescriptor_ALL;
    static const Interval SamplesInterval_ALL;

    SamplesIntervalDescriptor( );
    SamplesIntervalDescriptor( Interval );
    SamplesIntervalDescriptor( SampleType first, SampleType last );
    SamplesIntervalDescriptor operator | (const SamplesIntervalDescriptor& b) const { SamplesIntervalDescriptor a = *this; return a|=b; }
    SamplesIntervalDescriptor& operator |= (const SamplesIntervalDescriptor&);
    SamplesIntervalDescriptor& operator |= (const Interval&);
    SamplesIntervalDescriptor operator - (const SamplesIntervalDescriptor& b) const { SamplesIntervalDescriptor a = *this; return a-=b; }
    SamplesIntervalDescriptor& operator -= (const SamplesIntervalDescriptor&);
    SamplesIntervalDescriptor& operator -= (const Interval&);
    SamplesIntervalDescriptor& operator -= (const SampleType&);
    SamplesIntervalDescriptor& operator += (const SampleType&);
    SamplesIntervalDescriptor operator & (const SamplesIntervalDescriptor& b) const { SamplesIntervalDescriptor a = *this; return a&=b; }
    SamplesIntervalDescriptor& operator &= (const SamplesIntervalDescriptor&);
    SamplesIntervalDescriptor& operator &= (const Interval&);
    SamplesIntervalDescriptor& operator *= (const float& scale);

    bool                            isEmpty() const { return _intervals.empty(); }
    Interval                        getInterval( SampleType dt, SampleType center = SampleType_MIN ) const;
    Interval                        getInterval( Interval n = SamplesInterval_ALL ) const;
    Interval                        coveredInterval() const;
    const std::list<Interval>&      intervals() const { return _intervals; }

    void                            print( std::string title="" ) const;

private:
    std::list<Interval> _intervals;
};

std::ostream& operator<<( std::ostream& s, const SamplesIntervalDescriptor& i);
std::ostream& operator<<( std::ostream& s, const SamplesIntervalDescriptor::Interval& i);

} // namespace Signal

#endif // SAMPLESINTERVALDESCRIPTOR_H
