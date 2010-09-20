#include "buffersource.h"

// TODO remove
#ifdef _MSC_VER
//typedef __int64 __int64_t;
#else
//#include <stdint.h> // defines __int64_t which is expected by sndfile.h
#endif

/*
#include <sndfile.hh> // for reading various formats
#include <math.h>
#include "Statistics.h"
#include <stdexcept>
#include <iostream>
#include <sstream>

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/algorithm/string.hpp>
*/
using namespace std;

namespace Signal {


BufferSource::
        BufferSource( pBuffer waveform )
:    _waveform(waveform)
{
}


pBuffer BufferSource::
        read( const Interval& I )
{
    if (Intervals(I) & _waveform->getInterval())
        return _waveform;

    return zeros(I);
}

} // namespace Signal
