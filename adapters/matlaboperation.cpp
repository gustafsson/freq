#include "matlaboperation.h"
#include "hdf5.h"

#include <signal.h>
#include <sys/stat.h>
#include <sstream>
#include <string.h>
#include <stdio.h>
//#include <boost/algorithm/string.hpp>
#include <fstream>
#include <sys/types.h>
#include <time.h>

#if defined(__GNUC__)
    #include <unistd.h>
    #include <sys/time.h>
#elif defined(WIN32)
    #include <windows.h>
    #include <process.h>
#endif

#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "tfr/chunk.h"

#include <QProcess>

using namespace std;
using namespace Signal;
using namespace boost;
using namespace boost::posix_time;

//#define TIME_MatlabFunction
#define TIME_MatlabFunction if(0)

namespace Adapters {

MatlabFunction::
        MatlabFunction( std::string matlabFunction, float timeout )
:   _pid(0),
    _matlab_function(matlabFunction),
    _timeout( timeout )
{
    BOOST_ASSERT(!matlabFunction.empty());

    { // Set filenames
        stringstream ss;
        ss << matlabFunction << "." << hex << this << ".h5";
        _dataFile = ss.str();
        _resultFile = _dataFile + ".result.h5";
    }

    { // Start matlab/octave
        stringstream matlab_command, octave_command;
        matlab_command << "addpath('/usr/share/sonicawe');"
                << "sawe_filewatcher('" << _dataFile << "',@" << matlabFunction << ");";
        octave_command << "addpath('/usr/share/sonicawe');"
                << "sawe_filewatcher_oct('" << _dataFile << "',@" << matlabFunction << ");";

        QStringList matlab_args;
        matlab_args.push_back("-r");
        matlab_args.push_back(matlab_command.str().c_str());
        QStringList octave_args;
        octave_args.push_back("-qf");
        octave_args.push_back("--eval");
        octave_args.push_back(octave_command.str().c_str());

        _pid = new QProcess();
        _pid->setProcessChannelMode( QProcess::ForwardedChannels );
        _pid->start("matlab", matlab_args);
        _pid->waitForStarted();
        if (_pid->state() == QProcess::Running)
            return;

        TaskInfo("Couldn't start MATLAB, trying Octave instead");
        _pid->start("octave", octave_args);
        _pid->waitForStarted();
        if (_pid->state() == QProcess::Running)
            return;

        TaskInfo("Couldn't start Octave");
        delete _pid;
        _pid = 0;
        /*
#ifdef __GNUC__
        _pid = (void*)fork();

        if(0==_pid)
        {
            ::execlp("matlab","matlab", "-r", matlab_command.str().c_str(), NULL );
            TaskInfo("Couldn't start MATLAB, trying Octave instead");
            // apparently failed, try octave
            ::execlp("octave","octave", "-qf", "--eval", octave_command.str().c_str(), NULL );
            TaskInfo("Couldn't start Octave");
            // failed that to... will eventually time out
            exit(0);
        }
#elif defined(WIN32)
		_pid = (void*)_spawnlp(_P_NOWAIT, "matlab", 
                        "matlab", "-noFigureWindows", "-nojvm", "-nodesktop", "-nosplash", "-r", matlab_command.str().c_str(), NULL );

		if (0 == _pid)
		{
			// failed, try octave
            _pid = (void*)_spawnlp(_P_NOWAIT, "octave","octave", "-qf", "--eval", octave_command.str().c_str(), NULL );
	        // will eventually time out if this fails to
		}
#else
#error No implementation to spawn processes implemented for this platform/compiler.
#endif // __GNUC__*/
    }
}

MatlabFunction::
        ~MatlabFunction()
{
	kill();
}

std::string MatlabFunction::
        invokeAndWait( std::string source )
{
	if (0==_pid)
	{
        TIME_MatlabFunction TaskTimer tt("Matlab/octave failed, ignoring.");
		return "";
	}

    boost::scoped_ptr<TaskTimer> tt;
    TIME_MatlabFunction tt.reset( new TaskTimer("Waiting for matlab/octave."));

    ::remove(_resultFile.c_str());
    ::rename(source.c_str(), _dataFile.c_str());

    // Wait for result to be written
    struct stat dummy;

    time_duration timeout(0, 0, _timeout,fmod(_timeout,1.f));
    ptime start = second_clock::local_time();

    while (0!=stat( _resultFile.c_str(),&dummy))
    {
#ifdef __GNUC__
        usleep(10 * 1000); // Sleep in microseconds
#elif defined(WIN32)
        Sleep(10); // Sleep in ms
#else
#error TODO implement
#endif
        time_duration d = second_clock::local_time()-start;
        if (timeout < d)
		{
            abort(); // throws
		}
        if (d.total_seconds() > 3)
            TIME_MatlabFunction tt->partlyDone();
    }
#ifdef WIN32 // wait for slow file system to finish move
    Sleep(100);
#endif
    return _resultFile;
}

string MatlabFunction::
        getTempName()
{
    return _dataFile + "~";
}


std::string MatlabFunction::
        matlabFunction()
{
    return _matlab_function;
}


float MatlabFunction::
        timeout()
{
    return _timeout;
}


void MatlabFunction::
		kill()
{
    if (_pid)
    {
        TaskTimer tt("MatlabFunction killing MATLAB/Octave process");
        tt.partlyDone();
        _pid->terminate();
        tt.partlyDone();
        _pid->waitForFinished();
        tt.partlyDone();
        delete _pid;

        ::remove(_dataFile.c_str());
        ::remove(_resultFile.c_str());
        ::remove("octave-core");
    }
/*	if (_pid)
	{
		#ifdef __GNUC__
			::kill((pid_t)(unsigned long long)_pid, SIGINT);
		#elif defined(WIN32)
			TerminateProcess((HANDLE)_pid, 0);
		#else
			#error No implementation
		#endif
		_pid = 0;
    }*/
}

void MatlabFunction::
		abort()
{
	kill();
	throw std::invalid_argument("Timeout in MatlabFunction::invokeAndWait");
}

MatlabOperation::
        MatlabOperation( Signal::pOperation source, std::string matlabFunction )
:   OperationCache(source),
    _matlab(new MatlabFunction(matlabFunction, 4))
{
}

MatlabOperation::
        ~MatlabOperation()
{
    TaskInfo("~MatlabOperation");
}

pBuffer MatlabOperation::
        readRaw( const Interval& I )
{
    TaskTimer tt("MatlabOperation::read(%u,%u), count = %u", I.first, I.last, (Signal::IntervalType)I.count() );

    // just 'read()' might return the entire signal, which would be way to
    // slow to export in an interactive manner
    IntervalType support = sample_rate()*0.5;
    Interval J = Intervals(I).addedSupport( support );

    pBuffer b = source()->readFixedLength( J );

    string file = _matlab->getTempName();

    Hdf5Buffer::saveBuffer( file, *b );

    file = _matlab->invokeAndWait( file );

	if (file.empty())
		return b;

    pBuffer b2 = Hdf5Buffer::loadBuffer( file );

	::remove( file.c_str());

    return BufferSource( b2 ).readFixedLength( I );
}

void MatlabOperation::
        restart()
{
    std::string fn = _matlab->matlabFunction();
    float t = _matlab->timeout();

    _cache.clear();
    _matlab.reset();
    _matlab.reset( new MatlabFunction( fn, t ));
}

} // namespace Adapters
