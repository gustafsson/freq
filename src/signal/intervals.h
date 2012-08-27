#ifndef SAMPLESINTERVALDESCRIPTOR_H
#define SAMPLESINTERVALDESCRIPTOR_H

#include "sawe/sawedll.h"

#include <list>
#include <string>

namespace Signal {

typedef long long IntervalType;

/**
  Describes one discrete intervals. Always in the same sample rate as the
  signal they are referring to. So, one 'Interval' is given an including
  beginning 'first' and exclusive end 'last' in integers such that

     I = [first, last)

  */
class SaweDll Interval {
public:
    static const IntervalType IntervalType_MIN;
    static const IntervalType IntervalType_MAX;
    static const Interval Interval_ALL;

    Interval();

    /**
      Failes with assertion if first>last.
      */
    Interval( IntervalType first, IntervalType last );

    /**
      Describes the interval [first, last). That is, 'last' is excluded from
      the interval. The number of samples in the interval is computed by
      "last-first".

      It is up to the user to ensure the invariant relation first<=last.
      */
    IntervalType first, last;
    IntervalType count() const { return valid() ? last - first : 0; }

    bool valid() const { return first <= last; }
    Interval operator|(const Interval& r) const { Interval I(*this); return I|=r; }
    Interval& operator|=(const Interval& r);
    Interval operator&(const Interval& r) const { Interval I(*this); return I&=r; }
    Interval& operator&=(const Interval& r);
    bool operator==(const Interval& r) const;
    bool operator!=(const Interval& r) const;
    operator   bool() const { return 0 < count(); }

    std::string toString() const;
};

#ifdef _MSC_VER
#pragma warning (push)
// warning C4251: 'Signal::Intervals::base_' : class 'std::list<_Ty>' needs to
// have dll-interface to be used by clients of class 'Signal::Intervals'
//
// As long as the .dll is only used internally for testing, this is not a problem.
#pragma warning (disable:4251)
#endif

/**
  Describes a bunch of discrete intervals. Always in the same sample rate as the
  signal they are referring to. So, one 'Interval' is given an including
  beginning 'first' and exclusive end 'last' in integers such that

     I = [first, last)

  */
class SaweDll Intervals: private std::list<Interval>
{
    typedef std::list<Interval> base;
public:
    static const Intervals Intervals_ALL;

    Intervals( );
    Intervals( const Interval& );
    Intervals( IntervalType first, IntervalType last );
    Intervals  operator |  (const Intervals& b) const { return Intervals(*this)|=b; }
    Intervals& operator |= (const Intervals&);
    Intervals& operator |= (const Interval&);
    Intervals  operator -  (const Intervals& b) const { return Intervals(*this)-=b; }
    Intervals& operator -= (const Intervals&);
    Intervals& operator -= (const Interval&);
    Intervals  operator &  (const Intervals& b) const { return Intervals(*this)&=b; }
    Intervals& operator &= (const Intervals&);
    Intervals& operator &= (const Interval&);
    Intervals  operator ^  (const Intervals& b) const { return Intervals(*this)^=b; }
    Intervals& operator ^= (const Intervals&);
    Intervals operator >> (const IntervalType& b) const { return Intervals(*this)>>=b; }
    Intervals& operator >>= (const IntervalType&);
    Intervals operator << (const IntervalType& b) const { return Intervals(*this)<<=b; }
    Intervals& operator <<= (const IntervalType&);
    Intervals& operator *= (const float& scale);
    Intervals  operator ~  () const { return inverse(); }
    operator   bool        () const { return !empty(); }

    Intervals               inverse() const;
    Interval                fetchFirstInterval() const;
    Interval                fetchInterval( IntervalType preferred_size, IntervalType center = Interval::IntervalType_MIN ) const;
    Interval                spannedInterval() const;
    Intervals               enlarge( IntervalType dt ) const;
    Intervals               shrink( IntervalType dt ) const;
    IntervalType            count() const;
    bool                    testSample( IntervalType const &p) const;

    std::string             toString() const;

    // STL compliant container
    typedef base::const_iterator const_iterator;
    typedef base::const_iterator iterator; // doesn't allow arbitrary changes in the list but permits iterations
    iterator                begin() { return base::begin(); }
    const_iterator          begin() const { return base::begin(); }
    iterator                end() { return base::end(); }
    const_iterator          end() const { return base::end(); }
    bool                    empty() const { return base::empty(); }
    void                    clear() { base::clear(); }
    bool operator==         (const Intervals& b) const { return ((base&)*this)==b; }
    bool operator!=         (const Intervals& b) const { return ((base&)*this)!=b; }

private:
    base::iterator firstIntersecting( const Interval& b );
};

std::ostream& operator<< (std::ostream& o, const Interval& I);
std::ostream& operator<< (std::ostream& o, const Intervals& I);
Intervals  operator |  (const Interval& a, const Intervals& b);
Intervals  operator -  (const Interval& a, const Intervals& b);
Intervals  operator &  (const Interval& a, const Intervals& b);
Intervals  operator ^  (const Interval& a, const Intervals& b);

} // namespace Signal

#ifdef _MSC_VER
#pragma warning (pop)
#endif

#endif // SAMPLESINTERVALDESCRIPTOR_H
