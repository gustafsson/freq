#ifndef SAMPLESINTERVALDESCRIPTOR_H
#define SAMPLESINTERVALDESCRIPTOR_H

#include "sawe/sawedll.h"

#include <list>
#include <string>

namespace Signal {

typedef long long IntervalType;
typedef unsigned long long UnsignedIntervalType;


/**
  Describes one discrete interval. Always in the same sample rate as the
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

    /**
     * count() returns the number of elements between first and last or 0 if last>first.
     */
    UnsignedIntervalType count() const { return valid() ? (UnsignedIntervalType)(last - first): 0u; }

    bool        valid       () const { return first <= last; }
    Interval    spanned     (const Interval& r) const;
    Interval    operator&   (const Interval& r) const { Interval I(*this); return I&=r; }
    Interval&   operator&=  (const Interval& r);
    bool        operator==  (const Interval& r) const;
    bool        operator!=  (const Interval& r) const;
    bool        contains    (const Interval& t) const;
    bool        contains    (const IntervalType& t) const;
    operator    bool        () const { return first < last; } // == 0 < count()

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

  Could speed up performance by allocating a limited number of Interval on the
  stack instead, if performance of Intervals becomes an issue.
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
    // These are ambiguous due to 'operator bool' below. 'operator bool' is more commonly used.
    //Intervals  operator >> (const IntervalType& b) const { return Intervals(*this)>>=b; }
    //Intervals  operator << (const IntervalType& b) const { return Intervals(*this)<<=b; }
    Intervals& operator >>=(const IntervalType&);
    Intervals& operator <<=(const IntervalType&);
    Intervals& operator *= (const float& scale);
    Intervals  operator ~  () const { return inverse(); }
    operator   bool        () const { return !empty(); }

    // contains returns true only if the entire argument is covered by this
    bool                    contains    (const Intervals& t) const;
    bool                    contains    (const Interval& t) const;
    bool                    contains    (const IntervalType& t) const;
    Intervals               inverse() const;
    Interval                fetchFirstInterval() const;
    Interval                fetchInterval( UnsignedIntervalType preferred_size, IntervalType center = Interval::IntervalType_MIN ) const;
    Interval                spannedInterval() const;
    Intervals               enlarge( IntervalType dt ) const;
    Intervals               shrink( IntervalType dt ) const;
    IntervalType            count() const;
    int                     numSubIntervals() const { return base::size(); }
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

public:
    static void test();
};

SaweDll std::ostream& operator<< (std::ostream& o, const Interval& I);
SaweDll std::ostream& operator<< (std::ostream& o, const Intervals& I);
SaweDll Intervals  operator |  (const Interval& a, const Intervals& b);
SaweDll Intervals  operator -  (const Interval& a, const Intervals& b);
SaweDll Intervals  operator &  (const Interval& a, const Intervals& b);
SaweDll Intervals  operator ^  (const Interval& a, const Intervals& b);
SaweDll Intervals  operator |  (const Interval& a, const Interval& b);

} // namespace Signal

#ifdef _MSC_VER
#pragma warning (pop)
#endif

#endif // SAMPLESINTERVALDESCRIPTOR_H
