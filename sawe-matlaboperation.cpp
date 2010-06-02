#include "sawe-matlaboperation.h"
#include "sawe-hdf5.h"
#include <signal.h>
#include <sys/stat.h>
#include <sstream>
#include <string.h>
#include <stdio.h>
//#include <boost/algorithm/string.hpp>
#include <fstream>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "tfr-chunk.h"

using namespace std;
using namespace Signal;
using namespace boost;
using namespace boost::posix_time;

namespace Sawe {

MatlabFunction::
        MatlabFunction( std::string matlabFunction, float timeout )
:   _pid(0),
    _timeout( timeout )
{
    BOOST_ASSERT(!matlabFunction.empty());

    { // Set filenames
        stringstream ss;
        ss << matlabFunction << "." << hex << this << ".h5";
        _dataFile = ss.str();
        _resultFile = _dataFile + ".result.h5";
    }

    { // Start octave
        stringstream ss;
        ss << "filewatcher('" << _dataFile << "', @" << matlabFunction << ");";

        _pid = fork();

        if(0==_pid)
        {
            ::execlp("matlab","matlab", "-qf", "--eval", ss.str().c_str(), NULL );
            // apparently failed, try matlab
            ::execlp("octave","octave", "-qf", "--eval", ss.str().c_str(), NULL );
            // failed that to... will eventually time out
            exit(0);
        }
    }
}

MatlabFunction::
        ~MatlabFunction()
{
    kill((pid_t)_pid, SIGINT);
}

std::string MatlabFunction::
        invokeAndWait( std::string source )
{
    TaskTimer tt("Waiting for matlab/octave.");

    remove(_resultFile.c_str());
    rename(source.c_str(), _dataFile.c_str());

    // Wait for result to be written
    struct stat dummy;

    time_duration timeout(0,0,_timeout,fmod(_timeout,1.f));
    ptime start = second_clock::local_time();

    while (0!=stat( _resultFile.c_str(),&dummy))
    {
        usleep(10 * 1000); // Sleep for 10 ms
        time_duration d = second_clock::local_time()-start;
        if (timeout < d)
            throw std::runtime_error("Timeout in MatlabFunction::invokeAndWait");
        if (d.total_seconds() > 3)
            tt.partlyDone();
    }

    return _resultFile;
}

string MatlabFunction::
        getTempName()
{
    return _dataFile + "~";
}

MatlabOperation::
        MatlabOperation( Signal::pSource source, std::string matlabFunction )
:   OperationCache(source),
    _matlab(matlabFunction)
{
}

pBuffer MatlabOperation::
        readRaw( unsigned firstSample, unsigned numberOfSamples )
{
    TaskTimer tt("MatlabOperation::read(%u,%u)", firstSample, numberOfSamples );

    pBuffer b = _source->read( firstSample, numberOfSamples );

    string file = _matlab.getTempName();

    Hdf5Sink::saveBuffer( file, *b );

    file = _matlab.invokeAndWait( file );

    pBuffer b2 = Hdf5Sink::loadBuffer( file );
    b->waveform_data.swap( b2->waveform_data );

    ::remove( file.c_str());

    return b;
}

} // namespace Sawe
