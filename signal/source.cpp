#include "source.h"

#include <demangle.h>
#include <TaskTimer.h>

#include <sstream>
#include <iomanip>

//#define TIME_READCHECKED
#define TIME_READCHECKED if(0)

using namespace std;

namespace Signal {


pBuffer SourceBase::
        readChecked( const Interval& I )
{
    BOOST_ASSERT( I.count() );

    pBuffer r = read(I);

    // Check if read returned any samples form the interval I
    BOOST_ASSERT(r->sample_offset < I.last);
    BOOST_ASSERT(r->sample_offset + r->number_of_samples() > I.first);

    return r;
}

pBuffer SourceBase::
        readFixedLength( const Interval& I )
{
    TIME_READCHECKED TaskTimer tt("%s.%s %s",
                  vartype(*this).c_str(), __FUNCTION__ ,
                  I.toString().c_str() );

    // Try a simple read
    pBuffer p = readChecked( I );
    if (I == p->getInterval())
        return p;

    // This row gives some performance gain (cpu->gpu copy only once and never back until inverse).
    // But this also increases complexity to be handled properyl, that is not coded yet.
    //p->waveform_data()->getCudaGlobal();

    // Didn't get exact result, check if it spans all of I
    if ( (p->getInterval() & I) == I )
    {
        pBuffer r( new Buffer( I, p ));
        BOOST_ASSERT( r->getInterval() == I );
        return r;
    }

    // Doesn't span all of I, prepare new Buffer
    pBuffer r( new Buffer(I.first, I.count(), p->sample_rate ) );

//    if ( p->waveform_data()->getMemoryLocation() == GpuCpuVoidData::CudaGlobal )
//        r->waveform_data()->getCudaGlobal();

    Intervals sid(I);

    while (sid)
    {
        if (!p)
            p = readChecked( sid.fetchFirstInterval() );

        sid -= p->getInterval();
        (*r) |= *p; // Fill buffer
        p.reset();
    }

    return r;
}


string SourceBase::
        lengthLongFormat(float L)
{
    stringstream ss;
    unsigned seconds_per_minute = 60;
    unsigned seconds_per_hour = seconds_per_minute*60;
    unsigned seconds_per_day = seconds_per_hour*24;

    if (L < seconds_per_minute )
        ss << L << " s";
    else
    {
        if (L <= seconds_per_day )
        {
            unsigned days = floor(L/seconds_per_day);
            ss << days << "d ";
            L -= days * seconds_per_day;
        }

        unsigned hours = floor(L/seconds_per_hour);
        ss << setfill('0') << setw(2) << hours << ":";
        L -= hours * seconds_per_hour;

        unsigned minutes = floor(L/seconds_per_minute);
        ss << setfill('0') << setw(2) << minutes << ":";
        L -= minutes * seconds_per_minute;

        ss << setiosflags(ios::fixed)
           << setprecision(3) << setw(6) << L;
    }
    return ss.str();
}


pBuffer SourceBase::
        zeros( const Interval& I )
{
    BOOST_ASSERT( I.count() );

    TIME_READCHECKED TaskTimer tt("%s.%s %s",
                  vartype(*this).c_str(), __FUNCTION__ ,
                  I.toString().c_str() );

    pBuffer r( new Buffer(I.first, I.count(), sample_rate()) );
    memset(r->waveform_data()->getCpuMemory(), 0, r->waveform_data()->getSizeInBytes1D());
    return r;
}


} // namespace Signal
