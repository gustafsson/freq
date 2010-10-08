#ifndef SAMPLESINTERVALDESCRIPTOR_H
#define SAMPLESINTERVALDESCRIPTOR_H

#include <list>
#include <ostream>

namespace Signal {

typedef long unsigned IntervalType;

/**
  Describes one discrete intervals. Always in the same sample rate as the
  signal they are referring to. So, one 'Interval' is given an including
  beginning 'first' and exclusive end 'last' in integers such that

     I = [first, last)

  */
class Interval {
public:
    static const IntervalType IntervalType_MIN;
    static const IntervalType IntervalType_MAX;
    static const Interval Interval_ALL;

    Interval& operator=( const Interval& i)
                       { first = i.first, last = i.last; return *this; }
    Interval( IntervalType first, IntervalType last )
        :   first(first), last(last)
    {}

    /**
      'last' is rather the first sample not within the interval, such that
       the length of the interval can be computed as "last-first".
      */
    IntervalType first, last;
    IntervalType count() const { return last - first; }

    bool valid() const;
    bool isConnectedTo(const Interval& r) const;
    bool operator<(const Interval& r) const;
    Interval& operator|=(const Interval& r);
    bool operator==(const Interval& r) const;
};


/**
  Describes a bunch of discrete intervals. Always in the same sample rate as the
  signal they are referring to. So, one 'Interval' is given an including
  beginning 'first' and exclusive end 'last' in integers such that

     I = [first, last)

  */
class Intervals: public std::list<Interval>
{
public:
    static const Intervals Intervals_ALL;

    Intervals( );
    Intervals( const Interval& );
    Intervals( IntervalType first, IntervalType last );
    Intervals  operator |  (const Intervals& b) const { Intervals a = *this; return a|=b; }
    Intervals& operator |= (const Intervals&);
    Intervals& operator |= (const Interval&);
    Intervals  operator -  (const Intervals& b) const { Intervals a = *this; return a-=b; }
    Intervals& operator -= (const Intervals&);
    Intervals& operator -= (const Interval&);
    Intervals  operator &  (const Intervals& b) const { Intervals a = *this; return a&=b; }
    Intervals& operator &= (const Intervals&);
    Intervals& operator &= (const Interval&);
    Intervals& operator >>= (const IntervalType&);
    Intervals& operator <<= (const IntervalType&);
    Intervals& operator *= (const float& scale);
    Intervals  operator ~  () const { return inverse(); }
    operator   bool        () const { return !empty(); }

    Intervals                       inverse() const;

    Interval                        getInterval() const;
    Interval                        getInterval( IntervalType dt, IntervalType center = Interval::IntervalType_MIN ) const;
    Interval                        coveredInterval() const;

    void                            print( std::string title="" ) const;

};

std::ostream& operator<<( std::ostream& s, const Intervals& i);
std::ostream& operator<<( std::ostream& s, const Interval& i);

} // namespace Signal

#endif // SAMPLESINTERVALDESCRIPTOR_H
