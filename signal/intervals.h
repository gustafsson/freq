#ifndef SAMPLESINTERVALDESCRIPTOR_H
#define SAMPLESINTERVALDESCRIPTOR_H

#include <list>
#include <ostream>

namespace Signal {

/**
  Describes a bunch of discrete intervals. Always in the same sample rate as the
  signal they are referring to. So, one 'Interval' is given an including
  beginning 'first' and exclusive end 'last' in integers such that

     I = [first, last[

  */
class Intervals
{
public:
    typedef long unsigned SampleType;

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
    static const Intervals SamplesIntervalDescriptor_ALL;
    static const Interval SamplesInterval_ALL;

    Intervals( );
    Intervals( Interval );
    Intervals( SampleType first, SampleType last );
    Intervals operator | (const Intervals& b) const { Intervals a = *this; return a|=b; }
    Intervals& operator |= (const Intervals&);
    Intervals& operator |= (const Interval&);
    Intervals operator - (const Intervals& b) const { Intervals a = *this; return a-=b; }
    Intervals& operator -= (const Intervals&);
    Intervals& operator -= (const Interval&);
    Intervals& operator -= (const SampleType&);
    Intervals& operator += (const SampleType&);
    Intervals operator & (const Intervals& b) const { Intervals a = *this; return a&=b; }
    Intervals& operator &= (const Intervals&);
    Intervals& operator &= (const Interval&);
    Intervals& operator *= (const float& scale);

    bool                            isEmpty() const { return _intervals.empty(); }
    Interval                        getInterval( SampleType dt, SampleType center = SampleType_MIN ) const;
    Interval                        getInterval( Interval n = SamplesInterval_ALL ) const;
    Interval                        coveredInterval() const;
    const std::list<Interval>&      intervals() const { return _intervals; }

    void                            print( std::string title="" ) const;

private:
    std::list<Interval> _intervals;
};

std::ostream& operator<<( std::ostream& s, const Intervals& i);
std::ostream& operator<<( std::ostream& s, const Intervals::Interval& i);

} // namespace Signal

#endif // SAMPLESINTERVALDESCRIPTOR_H
