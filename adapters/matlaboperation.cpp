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
    std::string path = QFileInfo(f.c_str()).path().replace("'", "\\'") .toStdString();
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
                    << "sawe_filewatcher('" << _dataFile << "',@" << _matlab_function;
            octave_command
                    << "addpath('" << path << "');"
                    << "sawe_filewatcher('" << _dataFile << "',@" << _matlab_function;

            std::string arguments = settings->arguments();

            // remove trailing ';'
            while(arguments.size() && arguments[ arguments.size() - 1] == ';')
                arguments.resize( arguments.size() - 1);

            if (arguments.size())
            {
                matlab_command << ", " << arguments;
                octave_command << ", " << arguments;
            }

            matlab_command << ");";
            octave_command << ");";
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
    endProcess();
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
    if (!_pid)
        return;

    if (!hasProcessEnded())
        _pid->kill();  // send SIGKILL
    else
        _pid->terminate(); // send platform specific "please close message"

    delete _pid;
    _pid = 0;
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
		abort()
{
    endProcess();
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

    settings(0);
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
    TaskInfo("MatlabOperation invalidate_samples(%s)", I.toString().c_str());
    TaskInfo("MatlabOperation children: %s", toString().c_str());
    TaskInfo("MatlabOperation outputs: %s", parentsToString().c_str());

    if (_settings && _settings->computeInOrder() && (I - _cache.invalid_samples()))
    {
        // Start over and recompute the first block again
        TaskInfo("MatlabOperation start over");
        OperationCache::invalidate_samples(Signal::Intervals::Intervals_ALL);
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
        pBuffer plot_pts;
        ready_data = Hdf5Buffer::loadBuffer( file, &redundancy, &plot_pts );

        Interval J = ready_data->getInterval();

        if (this->plotlines){ // Update plot
            Tools::Support::PlotLines& plotlines = *this->plotlines.get();
            plotlines.clear( J, ready_data->sample_rate );

            if (plot_pts)
            {
                float start = ready_data->start();
                float length = ready_data->length();

                float* p = plot_pts->waveform_data()->getCpuMemory();
                cudaExtent N = plot_pts->waveform_data()->getNumberOfElements();
                unsigned id = this->get_channel();
                if (3 <= N.height)
                    for (unsigned x=0; x<N.width; ++x)
                        plotlines.set( id, p[ x ], p[ x + N.width ], p[ x + 2*N.width ] );
                else if (2 == N.height)
                    for (unsigned x=0; x<N.width; ++x)
                        plotlines.set( id, p[ x ], p[ x + N.width ] );
                else if (1 == N.height)
                    for (unsigned x=0; x<N.width; ++x)
                        plotlines.set( id, start + (x+0.5)*length/N.width, p[ x ] );
            }
        }

        ::remove( file.c_str());

        if (_settings->chunksize() < 0)
            redundancy = 0;

        IntervalType support = (IntervalType)std::floor(redundancy + 0.5);
        _settings->redundant(support);

        if ((invalid_samples() - J).empty())
            _matlab->endProcess(); // Finished with matlab

        TaskInfo("invalid_returns = %s, J = %s, invalid_returns & J = %s",
                 invalid_returns().toString().c_str(),
                 J.toString().c_str(),
                 (invalid_returns()&J).toString().c_str());

        invalidate_samples( invalid_returns() & J );

        return true;
    }

    return false;
}


bool MatlabOperation::
        isWaiting()
{
    return _matlab->isWaiting();
}


pBuffer MatlabOperation::
        readRaw( const Interval& I )
{
    if (!_matlab)
        return pBuffer();

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
            TaskInfo("Returning ready data %s, %u channels",
                     b->getInterval().toString().c_str(),
                     b->waveform_data()->getNumberOfElements().height );
            return b;
        }


        if (!isWaiting())
        {
            Signal::Interval J = I;
            IntervalType support = 0;

            if (_settings->chunksize() < 0)
                J = Interval(0, number_of_samples());
            else
            {
                if (_settings->computeInOrder() )
                    J = _cache.invalid_samples_current_channel().fetchInterval( I.count() );

                if (0<_settings->chunksize())
                    J.last = J.first + _settings->chunksize();
            }

            support = _settings->redundant();
            Intervals J2 = J;
            J = Intervals(J).enlarge( support );

            // just 'read()' might return the entire signal, which would be way to
            // slow to export in an interactive manner
            Signal::pBuffer b;
            unsigned C = num_channels();
            if (1 == C)
                b = source()->readFixedLength( J );
            else
            {
                unsigned current_channel = this->get_channel();
                b.reset( new Signal::Buffer(J.first, J.count(), sample_rate(), C ));

                float* dst = b->waveform_data()->getCpuMemory();
                for (unsigned i=0; i<C; ++i)
                {
                    source()->set_channel( i );
                    Signal::pBuffer r = source()->readFixedLength( J );
                    float* src = r->waveform_data()->getCpuMemory();
                    memcpy( dst + i*J.count(), src, J.count()*sizeof(float));
                }
                source()->set_channel( current_channel );
            }

            string file = _matlab->getTempName();

            Hdf5Buffer::saveBuffer( file, *b, support );

            TaskInfo("Sending %s to Matlab/Octave", b->getInterval().toString().c_str() );
            _matlab->invoke( file );
        }
        else
        {
            TaskInfo("Is waiting for Matlab/Octave to finish");
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
    if (_settings)
    {
        _matlab.reset( new MatlabFunction( _settings->scriptname(), 4, _settings ));

        if (source())
            invalidate_samples( getInterval() );
    }
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

    if (_settings)
    {
        _settings->operation = this;
    }

    restart();
}

} // namespace Adapters
