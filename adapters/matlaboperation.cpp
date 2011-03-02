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
#include <QFileInfo>

using namespace std;
using namespace Signal;
using namespace boost;
using namespace boost::posix_time;

//#define TIME_MatlabFunction
#define TIME_MatlabFunction if(0)

namespace Adapters {

MatlabFunction::
        MatlabFunction( std::string f, float timeout, MatlabFunctionSettings* settings )
:   _pid(0),
    _matlab_function(f),
    _matlab_filename(f),
    _timeout( timeout )
{
    std::string path = QFileInfo(f.c_str()).path().toStdString();
    _matlab_filename = QFileInfo(f.c_str()).fileName().toStdString();
    _matlab_function = QFileInfo(f.c_str()).baseName().toStdString();

    { // Set filenames
        stringstream ss;
        ss << _matlab_function << "." << hex << this << ".h5";
        _dataFile = ss.str();
        _resultFile = _dataFile + ".result.h5";
    }

    { // Start matlab/octave
        stringstream matlab_command, octave_command;
        matlab_command
                << "addpath('/usr/share/sonicawe');";
        octave_command
                << "addpath('/usr/share/sonicawe');";

        if (f.empty())
        {

        }
        else
        {
            matlab_command
                    << "addpath('" << path << "');"
                    << "sawe_filewatcher('" << _dataFile << "',@" << _matlab_function << ");";
            octave_command
                    << "addpath('" << path << "');"
                    << "sawe_filewatcher('" << _dataFile << "',@" << _matlab_function << ");";
        }

        QStringList matlab_args;
        // "-noFigureWindows", "-nojvm", "-nodesktop", "-nosplash"
        QStringList octave_args;
        octave_args.push_back("-qf");

        if (f.empty())
        {
            octave_args.push_back("--interactive");
        }
        else
        {
            matlab_args.push_back("-r");
            matlab_args.push_back(matlab_command.str().c_str());
            octave_args.push_back("--eval");
            octave_args.push_back(octave_command.str().c_str());
        }

        _pid = new QProcess();
//        _pid->setProcessChannelMode( QProcess::ForwardedChannels );
        _pid->setProcessChannelMode( QProcess::MergedChannels );
        if (settings) settings->setProcess( _pid );

        /*_pid->start("matlab", matlab_args);
        _pid->waitForStarted();
        if (_pid->state() == QProcess::Running)
            return;

        TaskInfo("Couldn't start MATLAB, trying Octave instead");*/

        _pid->start("octave", octave_args);
        _pid->waitForStarted();
        if (_pid->state() == QProcess::Running)
            return;

        TaskInfo("Couldn't start Octave");
        delete _pid;
        _pid = 0;
        /*
#if defined(WIN32)
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
#endif */
    }
}

MatlabFunction::
        ~MatlabFunction()
{
	kill();
}


string MatlabFunction::
        getTempName()
{
    return _dataFile + "~";
}


void MatlabFunction::
        invoke( std::string source )
{
    if (0==_pid)
    {
        TIME_MatlabFunction TaskTimer tt("Matlab/octave failed, ignoring.");
        return;
    }

    TIME_MatlabFunction TaskInfo("Invoking matlab/octave.");

    ::remove(_resultFile.c_str());
    ::rename(source.c_str(), _dataFile.c_str());
}


bool MatlabFunction::
        isWaiting()
{
    struct stat dummy;
    return 0==_pid || (isReady().empty() && 0==stat( _dataFile.c_str(),&dummy));
}


std::string MatlabFunction::
        isReady()
{
    struct stat dummy;
    if ( 0!=_pid && 0==stat( _resultFile.c_str(),&dummy) )
        return _resultFile;
    return "";
}


std::string MatlabFunction::
        waitForReady()
{
    boost::scoped_ptr<TaskTimer> tt;
    TIME_MatlabFunction tt.reset( new TaskTimer("Waiting for matlab/octave."));

    // Wait for result to be written
    time_duration timeout(0, 0, _timeout,fmod(_timeout,1.f));
    ptime start = second_clock::local_time();

    while (isReady().empty())
    {
#ifdef __GNUC__
        usleep(10 * 1000); // Sleep in microseconds
#elif defined(WIN32)
        Sleep(10); // Sleep in ms
#else
#error Does not support this platform
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


bool MatlabFunction::
        hasProcessEnded()
{
    return !_pid || _pid->state() == QProcess::NotRunning;
}


void MatlabFunction::
        endProcess()
{
    if (!hasProcessEnded())
        _pid->kill();  // send SIGKILL
    else
        _pid->terminate(); // send platform specific "please close message"
}


std::string MatlabFunction::
        matlabFunction()
{
    return _matlab_function;
}


std::string MatlabFunction::
        matlabFunctionFilename()
{
    return _matlab_filename;
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

        _pid = 0;
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
        MatlabOperation( Signal::pOperation source, MatlabFunctionSettings* settings )
:   OperationCache(source),
    _matlab(new MatlabFunction(settings->scriptname(), 4, settings)),
    _settings(settings)
{
}


MatlabOperation::
        MatlabOperation()
:   OperationCache(Signal::pOperation()),
    _settings(0)
{
}



MatlabOperation::
        ~MatlabOperation()
{
    TaskInfo("~MatlabOperation");
    TaskInfo(".");
    _matlab->endProcess();
    _matlab.reset();

    if (_settings->operation)
    {
        _settings->operation = 0;
        delete _settings;
    }
}


std::string MatlabOperation::
        name()
{
    if (!_matlab)
        return Operation::name();
    return _matlab->matlabFunctionFilename();
}


void MatlabOperation::
        invalidate_samples(const Intervals& I)
{
    // If computing in order and invalidating something that has already been
    // computed
    TaskInfo("MatlabOperation children: %s", toString().c_str());
    TaskInfo("MatlabOperation outputs: %s", parentsToString().c_str());

    if (_settings->computeInOrder() && (I - _cache.invalid_samples()))
    {
        // Start over and recompute the first block again
        OperationCache::invalidate_samples(getInterval());
    }
    else
        OperationCache::invalidate_samples(I);
}


bool MatlabOperation::
        dataAvailable()
{
    if (ready_data)
        return true;

    std::string file = _matlab->isReady();
    if (!file.empty())
    {
        double redundancy=0;
        ready_data = Hdf5Buffer::loadBuffer( file, &redundancy );

        ::remove( file.c_str());

        if (_settings->chunksize() < 0)
            redundancy = 0;

        IntervalType support = (IntervalType)std::floor(redundancy + 0.5);
        _settings->redundant(support);

        Interval J = ready_data->getInterval();

        if ((invalid_samples() - J).empty())
            _matlab->endProcess(); // Finished with matlab

        invalidate_samples( invalid_returns() );

        return true;
    }

    return false;
}

pBuffer MatlabOperation::
        readRaw( const Interval& I )
{
    if (_matlab->hasProcessEnded())
    {
        TaskInfo("MatlabOperation::read(%s), process ended", I.toString().c_str());
        return source()->read( I );
    }

    TaskTimer tt("MatlabOperation::read(%s)", I.toString().c_str() );

    try
    {
        if (dataAvailable())
        {
            Signal::pBuffer b = ready_data;
            ready_data.reset();
            return b;
        }


        if (!_matlab->isWaiting())
        {
            Signal::Interval J = I;
            IntervalType support = 0;

            if (_settings->chunksize() < 0)
                J = Interval(0, number_of_samples());
            else
            {
                if (_settings->computeInOrder() )
                    J = invalid_samples().fetchInterval( I.count() );

                if (0<_settings->chunksize())
                    J.last = J.first + _settings->chunksize();
            }

            support = _settings->redundant();
            Signal::Interval R = J;
            J = Intervals(J).enlarge( support );

            // just 'read()' might return the entire signal, which would be way to
            // slow to export in an interactive manner
            Signal::pBuffer b = source()->readFixedLength( J );

            string file = _matlab->getTempName();

            Hdf5Buffer::saveBuffer( file, *b, support );

            _matlab->invoke( file );
        }
    }
    catch (const std::runtime_error& e)
    {
        TaskInfo("MatlabOperation caught %s", e.what());
        _matlab->endProcess();
        throw std::invalid_argument( e.what() ); // invalid_argument doesn't crash the application
    }

    return pBuffer();
}


void MatlabOperation::
        restart()
{
    _cache.clear();
    _matlab.reset();
    _matlab.reset( new MatlabFunction( _settings->scriptname(), 4, _settings ));

    if (source())
        invalidate_samples( getInterval() );
}


void MatlabOperation::
        settings(MatlabFunctionSettings* settings)
{
    if (_settings && _settings->operation)
    {
        _settings->operation = 0;
        delete _settings;
    }

    _settings = settings;
    restart();
}

} // namespace Adapters
