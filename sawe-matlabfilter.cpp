#include "sawe-matlabfilter.h"
#include "sawe-hdf5.h"
#include <signal.h>
#include <sys/stat.h>
#include <sstream>
#include <fstream>
#include <string.h>
#include <stdio.h>

using namespace std;
using namespace Signal;
using namespace Tfr;

namespace Sawe {

MatlabFilter::
        MatlabFilter( std::string matlabFunction )
:   _matlab(matlabFunction)
{
}

void MatlabFilter::
        operator()( Chunk& c)
{
    TaskTimer tt("MatlabFilter::operator()(%f,%f)", c.startTime(), c.endTime() );

    string file = _matlab.getTempName();

    Hdf5Sink::saveChunk( file, c );

    file = _matlab.invokeAndWait( file );

    Tfr::pChunk pc = Hdf5Sink::loadChunk( file );
    c.transform_data.swap( pc->transform_data );

    ::remove( file.c_str());
}

Signal::SamplesIntervalDescriptor MatlabFilter::
        getZeroSamples( unsigned /*FS*/ ) const
{
    // As far as we know, the matlab filter doesn't set anything to zero for sure
    return Signal::SamplesIntervalDescriptor();
}

Signal::SamplesIntervalDescriptor MatlabFilter::
        getUntouchedSamples( unsigned /*FS*/ ) const
{
    // As far as we know, the matlab filter may touch anything
    return Signal::SamplesIntervalDescriptor();
}


} // namespace Sawe
