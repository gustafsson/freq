#ifndef SAMPLESINTERVALDESCRIPTOR_H
#define SAMPLESINTERVALDESCRIPTOR_H

#include <list>
#include <string>

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
      Describes the interval [first, last). That is, 'last' is excluded from
      the interval. The length of the interval is computed by "last-first".
      */
    IntervalType first, last;
    IntervalType count() const { return last - first; }

    bool valid() const;
    bool isConnectedTo(const Interval& r) const;
    bool operator<(const Interval& r) const;
    Interval& operator|=(const Interval& r);
    bool operator==(const Interval& r) const;
    bool operator!=(const Interval& r) const;

    std::string toString() const;
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
    Intervals operator >> (const IntervalType& b) const { Intervals a = *this; return a>>=b; }
    Intervals& operator >>= (const IntervalType&);
    Intervals operator << (const IntervalType& b) const { Intervals a = *this; return a<<=b; }
    Intervals& operator <<= (const IntervalType&);
    Intervals& operator *= (const float& scale);
    Intervals  operator ~  () const { return inverse(); }
    operator   Interval    () const { return coveredInterval(); }
    operator   bool        () const { return !empty(); }

    Intervals                       inverse() const;
    Interval                        getInterval() const;
    Interval                        getInterval( IntervalType dt, IntervalType center = Interval::IntervalType_MIN ) const;
    Interval                        coveredInterval() const;
    Intervals                       addedSupport( IntervalType dt ) const;
    IntervalType                    count() const;

    std::string                     toString() const;
};

} // namespace Signal

#endif // SAMPLESINTERVALDESCRIPTOR_H
