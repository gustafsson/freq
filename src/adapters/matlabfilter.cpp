#include "matlabfilter.h"
#include "hdf5.h"

#include "tfr/chunk.h"
#include "signal/computingengine.h"

#include <signal.h>
#include <sys/stat.h>
#include <sstream>
#include <fstream>
#include <string.h>
#include <stdio.h>

using namespace std;
using namespace Signal;
using namespace Tfr;

//#define TIME_MatlabFilter
#define TIME_MatlabFilter if(0)

namespace Adapters {

class ComputingMatlab: public Signal::ComputingEngine {};


MatlabFilterKernelDesc::
    MatlabFilterKernelDesc(std::string matlabFunction)
    :
      matlabFunction(matlabFunction)
{}


Tfr::pChunkFilter MatlabFilterKernelDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    if (dynamic_cast<ComputingMatlab*>(engine))
        return Tfr::pChunkFilter(new MatlabFilter(matlabFunction));

    return Tfr::pChunkFilter();
}


MatlabFilter::
        MatlabFilter( std::string matlabFunction )
:   _matlab(new MatlabFunction(matlabFunction, 15, 0))
{
}


void MatlabFilter::
        operator()( ChunkAndInverse& chunk )
{
    Chunk& c = *chunk.chunk.get ();
    TIME_MatlabFilter TaskTimer tt("MatlabFilter::operator() [%g,%g)", c.startTime(), c.endTime() );

    _invalid_returns |= c.getInterval();

    std::string file = _matlab->isReady();
    if (!file.empty())
    {
        Tfr::pChunk pc = Hdf5Chunk::loadChunk( file );
        c.transform_data.swap( pc->transform_data );

        ::remove( file.c_str());

        Interval J = c.getInterval();

        // TODO deprecateCache
        //DeprecatedOperation::invalidate_samples( _invalid_returns & J );
        _invalid_returns -= J;

        // TODO Perform inverse
        //return true;
        return;
    }

    if (!_matlab->isWaiting())
    {
        string file = _matlab->getTempName();

        Hdf5Chunk::saveChunk( file, c );

        _matlab->invoke( file );
    }

    // TODO Don't perform inverse. Tfr::ChunkFilter::NoInverseTag
    // Suggested: In general, compute an inverse but in this case compute a dummy inverse and return that instead.
    //return false;
}


void MatlabFilter::
        restart()
{
    std::string fn = _matlab->matlabFunction();
    float t = _matlab->timeout();

    _matlab.reset();
    _matlab.reset( new MatlabFunction( fn, t, 0 ));
}

} // namespace Adapters

namespace Adapters {

void MatlabFilterKernelDesc::
        test()
{
    // Can't test this as it requires an external process to be launched.
    // Or is that ok? Backtrace executes an external process...
    EXCEPTION_ASSERT(false);
}

} // namespace Adapters
