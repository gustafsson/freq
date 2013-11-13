#include "cpuproperties.h"

#include "TaskTimer.h"

#include <boost/date_time/posix_time/posix_time.hpp>


double CpuProperties::
        cpu_memory_speed(unsigned *sz)
{
    void *a, *b;
    unsigned n = (1<<26);
    unsigned M = 10;
    if (sz)
        *sz = 2*M*n;
    a = malloc( n );
    b = malloc( n );
    memset( a, 0x3e, n );
    memcpy( b, a, n ); // cold run
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    for (unsigned m=0; m<M; ++m)
        memcpy( b, a, n ); // warm run
    boost::posix_time::time_duration d = boost::posix_time::microsec_clock::local_time() - start;
    double dt = d.total_microseconds()*1e-6;

    free( b );
    free( a );

    return 2*M*n/dt;
}
