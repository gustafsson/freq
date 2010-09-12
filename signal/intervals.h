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

     I = [first, last[

  */
class Interval {
public:
    static const IntervalType IntervalType_MIN;
    static const IntervalType IntervalType_MAX;
    static const Interval Interval_ALL;

    Interval& operator=( const Interval& i)
                       { first = i.first, last = i.last; return *this; }
    Interval( IntervalType first, IntervalType last )
        :   first(first), last(last), count(*this)
    {}

    /**
      'last' is rather the first sample not within the interval, such that
       the length of the interval can be computed as "last-first".
      */
    IntervalType first, last;

    class Count { public: Count( Interval& i ): i(i) {}
        operator IntervalType() const
        {
            return i.last - i.first;
        }
    private: Interval& i;
    } count; // I really wanted this to behave like a read-only property

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

     I = [first, last[

  */
class Intervals
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
    Intervals& operator -= (const IntervalType&);
    Intervals& operator += (const IntervalType&);
    Intervals  operator &  (const Intervals& b) const { Intervals a = *this; return a&=b; }
    Intervals& operator &= (const Intervals&);
    Intervals& operator &= (const Interval&);
    Intervals& operator *= (const float& scale);
    Intervals  operator ~  () const { return inverse(); }
    operator   bool        () const { return !isEmpty(); }

    Intervals                       inverse() const;

    bool                            isEmpty() const { return _intervals.empty(); }
    Interval                        getInterval( IntervalType dt, IntervalType center = Interval::IntervalType_MIN ) const;
    Interval                        getInterval( Interval n = Interval::Interval_ALL ) const;
    Interval                        coveredInterval() const;
    const std::list<Interval>&      intervals() const { return _intervals; }

    void                            print( std::string title="" ) const;

private:
    std::list<Interval> _intervals;
};

std::ostream& operator<<( std::ostream& s, const Intervals& i);
std::ostream& operator<<( std::ostream& s, const Interval& i);

} // namespace Signal

#endif // SAMPLESINTERVALDESCRIPTOR_H
