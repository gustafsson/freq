#include "matlaboperation.h"
#include "hdf5.h"
#include "microphonerecorder.h"

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
#include <QApplication>
#include <QErrorMessage>
#include <QDir>

using namespace std;
using namespace Signal;
using namespace boost;
using namespace boost::posix_time;

//#define TIME_MatlabFunction
#define TIME_MatlabFunction if(0)

namespace Adapters {

bool startProcess(QProcess* pid, const QString& name, const QStringList& args)
{
    TaskInfo ti("Trying: ");
    ti.tt().getStream() << "\"" << name.toStdString() << "\" ";
    foreach (QString a, args)
        ti.tt().getStream() << "\"" << a.toStdString() << "\" ";
    ti.tt().flushStream();

    pid->start(name, args);
    pid->waitForStarted();
    if (pid->state() == QProcess::Running)
        return true;
    return false;
}


bool startProcess(QProcess* pid, const QStringList& names, const QStringList& args)
{
    foreach( QString name, names)
    {
        if (startProcess(pid, name, args))
            return true;
    }
    return false;
}


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

        {
            QTemporaryFile tempFile;
            tempFile.open();
            ss << tempFile.fileName().toStdString();
        }

        _dataFile = ss.str() + ".h5";
        _resultFile = _dataFile + ".result.h5";
    }

    { // Start matlab/octave
        stringstream matlab_command, octave_command;
        QString sawescript_paths[] =
        {
            // local working directory
            "matlab",
#if defined(_WIN32) || defined(__APPLE__)
            // windows and mac install path
            QApplication::applicationDirPath().replace("\\", "\\\\").replace("\'", "\\'" ) + "/matlab",
#else
            // ubuntu
            "/usr/share/sonicawe",
#endif
        };

        string scriptpath;
        for (unsigned i=0; i<sizeof(sawescript_paths)/sizeof(sawescript_paths[0]); i++)
            if (QDir(sawescript_paths[i]).exists())
            {
                scriptpath = sawescript_paths[i].toStdString();
                break;
            }

        if (!scriptpath.empty())
        {
            matlab_command
                    << "addpath('" << scriptpath << "');";
            octave_command
                    << "addpath('" << scriptpath << "');";
        }
        else
        {
            QErrorMessage::qtHandler()->showMessage("Couldn't locate required Sonic AWE scripts");
        }

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
        //matlab_args.push_back("-noFigureWindows");
        //matlab_args.push_back("-nojvm");
        matlab_args.push_back("-nodesktop");
        //matlab_args.push_back("-nosplash");
        QStringList octave_args;
        octave_args.push_back("-qf");

        if (f.empty())
        {
            octave_args.push_back("--interactive");
        }
        else
        {
            matlab_args.push_back("-r");
            matlab_args.push_back(QString::fromStdString(matlab_command.str()));
            octave_args.push_back("--eval");
            octave_args.push_back(QString::fromStdString(octave_command.str()));
        }

        _pid = new QProcess();
        connect( _pid, SIGNAL(finished( int , QProcess::ExitStatus )), SLOT(finished(int,QProcess::ExitStatus)));
//        _pid->setProcessChannelMode( QProcess::ForwardedChannels );
        _pid->setProcessChannelMode( QProcess::MergedChannels );
        if (settings) settings->setProcess( _pid );

        if (!f.empty())
        {
            if (startProcess(_pid, "matlab", matlab_args))
                return;

            TaskInfo("Couldn't start MATLAB, trying Octave instead");
        }

        QStringList octave_names;
        octave_names.push_back("octave-3.2.3");
        octave_names.push_back("octave");
        if (startProcess(_pid, octave_names, octave_args))
            return;

        TaskInfo("Couldn't start Octave");

        if (!f.empty())
        {
#ifdef _WIN32
            TaskInfo("Trying common installation paths for MATLAB instead");
            QStringList matlab_paths;
            matlab_paths.push_back("C:\\Program Files\\MATLAB\\R2008b\\bin\\matlab.exe");
            matlab_paths.push_back("C:\\Program Files (x86)\\MATLAB\\R2008b\\bin\\matlab.exe");

            if (startProcess(_pid, matlab_paths, matlab_args))
                return;

            TaskInfo("Couldn't start Matlab");
#endif
        }

#if defined(_WIN32) || defined(__APPLE__)
        TaskInfo("Trying common installation paths for Octave instead");

        QStringList octave_paths;
#ifdef _WIN32
        octave_paths.push_back("C:\\Octave\\3.2.3_gcc-4.4.0\\bin\\octave-3.2.3.exe");
#endif
#ifdef __APPLE__
        octave_paths.push_back("/Applications/Octave.app/Contents/Resources/bin/octave");
#endif
        if (startProcess(_pid, octave_paths, octave_args))
            return;

        TaskInfo("Couldn't find Matlab nor Octave");
#endif
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
		finished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (exitCode != 0)
    {
        _hasCrashed = TRUE;
    }
    else
    {
        _hasCrashed = FALSE;
    }
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
    {
        ifstream t(_resultFile.c_str());
        if (t.is_open())
            return _resultFile;
    }
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

    return _resultFile;
}


bool MatlabFunction::
        hasProcessEnded()
{
    return !_pid || _pid->state() == QProcess::NotRunning;
}

bool MatlabFunction::
        hasProcessCrashed()
{
    return _hasCrashed;
}

void MatlabFunction::
        endProcess()
{
    if (!_pid)
        return;

    //if (!hasProcessEnded())
    _pid->kill();  // send SIGKILL
    //else
    //    _pid->terminate(); // send platform specific "please close message"

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
        MatlabOperation( Signal::pOperation source, MatlabFunctionSettings* s )
:   OperationCache(source),
    _settings(0)
{
    settings(s);
}


MatlabOperation::
        MatlabOperation()
:   OperationCache(Signal::pOperation()),
    _settings(0)
{
    settings(0);
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


std::string MatlabOperation::
        functionName()
{
    if (!_matlab)
        return Operation::name();
    return _matlab->matlabFunction();
}


void MatlabOperation::
        invalidate_samples(const Intervals& I)
{
    // If computing in order and invalidating something that has already been
    // computed
    TaskInfo("MatlabOperation invalidate_samples(%s)", I.toString().c_str());

    Intervals previously_computed = cached_samples() & ~invalid_returns();
    bool start_over = _settings && _settings->computeInOrder() && (I & previously_computed);

    if (start_over)
    {
        // Start over and recompute all blocks again
        restart();
    }
    else
    {
        OperationCache::invalidate_samples( I );

        if (plotlines && source())
            plotlines->clear( I, sample_rate() );
    }
}


bool MatlabOperation::
        dataAvailable()
{
    if (ready_data)
        return true;

    std::string file = _matlab->isReady();
    if (!file.empty())
    {
        TaskTimer tt("Reading data from Matlab/Octave");
        double redundancy=0;
        pBuffer plot_pts;

        try
        {
            ready_data = Hdf5Buffer::loadBuffer( file, &redundancy, &plot_pts );
        }
        catch (const Hdf5Error& e)
        {
            if (Hdf5Error::Type_OpenFailed == e.type() && e.data() == file)
            {
                // Couldn't open it for reading yet, wait
                return false;
            }

            throw e;
        }

        ::remove( file.c_str());

        if (_settings->chunksize() < 0)
            redundancy = 0;

        IntervalType support = (IntervalType)std::floor(redundancy + 0.5);
        _settings->redundant(support);

        if (!ready_data)
        {
            TaskInfo("Couldn't read data from Matlab/Octave");
            return false;
        }

        if (this->plotlines){ // Update plot
            Tools::Support::PlotLines& plotlines = *this->plotlines.get();

            if (plot_pts)
            {
                float start = ready_data->start();
                float length = ready_data->length();

                cudaExtent N = plot_pts->waveform_data()->getNumberOfElements();
                for (unsigned id=0; id<N.depth; ++id)
                {
                    float* p = plot_pts->waveform_data()->getCpuMemory() + id*N.width*N.height;

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
        }

        Interval oldI = sent_data->getInterval();
        Interval newI = ready_data->getInterval();

        float *oldP = sent_data->waveform_data()->getCpuMemory();
        float *newP = ready_data->waveform_data()->getCpuMemory();

        Intervals J;

        for (unsigned c=0; c<ready_data->channels() && c<sent_data->channels(); c++)
        {
            Interval equal = oldI & newI;
            oldP += equal.first - oldI.first;
            newP += equal.first - newI.first;
            oldP += oldI.count() * c;
            newP += newI.count() * c;
            for (unsigned i=0; i<equal.count();i++)
                if (*oldP != *newP)
                {
                    equal.last = equal.first;
                    break;
                }

            if (equal.count())
                _invalid_returns[c] -= equal;

            J |= newI - equal;
        }

        Signal::Intervals samples_to_invalidate = invalid_returns() & J;
        TaskInfo("invalid_returns = %s, J = %s, invalid_returns & J = %s",
                 invalid_returns().toString().c_str(),
                 J.toString().c_str(),
                 samples_to_invalidate.toString().c_str());

        if (J.empty())
        {
            TaskInfo("Matlab script didn't change anything");
        }
        else
        {
            TaskInfo("Matlab script made some changes");
        }

        if (samples_to_invalidate)
            OperationCache::invalidate_samples( samples_to_invalidate );

        MicrophoneRecorder* recorder = dynamic_cast<MicrophoneRecorder*>(root());
        bool isrecording = 0!=recorder;
        if (isrecording)
        {
            // Leave the process running so that we can continue a recording or change the list of operations
        }
        else
        {
            if (((invalid_samples() | invalid_returns()) - J).empty())
                _matlab->endProcess(); // Finished with matlab
        }

        return true;
    }

    return false;
}


bool MatlabOperation::
        isWaiting()
{
    return _matlab->isWaiting();
}


Interval MatlabOperation::
        intervalToCompute( const Interval& I )
{
    if (0 == I.count())
        return I;

    Signal::Interval J = I;

    if (_settings->chunksize() < 0)
        J = Interval(0, number_of_samples());
    else
    {
        if (_settings->computeInOrder() )
            J = (invalid_samples() | invalid_returns()).fetchInterval( I.count() );
        else
            J = (invalid_samples() | invalid_returns()).fetchInterval( I.count(), I.first );

        if (0<_settings->chunksize())
            J.last = J.first + _settings->chunksize();
    }

    IntervalType support = _settings->redundant();
    Interval signal = getInterval();
    J &= signal;
    Interval K = Intervals(J).enlarge( support ).coveredInterval();

    bool need_data_after_end = K.last > signal.last;
    if (0<_settings->chunksize() && (int)J.count() != _settings->chunksize())
        need_data_after_end = true;

    if (need_data_after_end)
    {
        MicrophoneRecorder* recorder = dynamic_cast<MicrophoneRecorder*>(root());
        bool isrecording = 0!=recorder;
        if (isrecording)
        {
            bool need_a_specific_chunk_size = 0<_settings->chunksize();
            if (need_a_specific_chunk_size)
            {
                if (recorder->isStopped() && !_settings->computeInOrder())
                {
                    // Ok, go on
                }
                else
                {
                    return Interval(0,0);
                }
            }
            else
            {
                if (recorder->isStopped())
                {
                    // Ok, go on
                }
                else
                {
                    return Interval(0,0);
                    // Don't use any samples after the end while recording
                    K &= signal;

                    if (Intervals(K).shrink(support).empty())
                        return Interval(0,0);
                }
            }
        }
    }

    return K;
}


pBuffer MatlabOperation::
        readRaw( const Interval& I )
{
    if (!_matlab)
        return pBuffer();

    try
    {
        if (dataAvailable())
        {
            Signal::pBuffer b = ready_data;
            ready_data.reset();
            TaskInfo("MatlabOperation::read(%s) Returning ready data %s, %u channels",
                     I.toString().c_str(),
                     b->getInterval().toString().c_str(),
                     b->waveform_data()->getNumberOfElements().height );
            return b;
        }

        if (_matlab->hasProcessEnded())
        {
            if (_matlab->hasProcessCrashed())
            {
                TaskInfo("MatlabOperation::read(%s) process ended", I.toString().c_str() );

                return source()->readFixedLength( I );
            }
            else
            {
                restart();
            }
        }

        if (!isWaiting())
        {
            TaskTimer tt("MatlabOperation::read(%s)", I.toString().c_str() );
            Interval K = intervalToCompute(I);

            if (0 == K.count())
                return pBuffer();

            // just 'read()' might return the entire signal, which would be way to
            // slow to export in an interactive manner
            sent_data = source()->readFixedLengthAllChannels( K );

            string file = _matlab->getTempName();

            IntervalType support = _settings->redundant();
            Hdf5Buffer::saveBuffer( file, *sent_data, support );

            TaskInfo("Sending %s to Matlab/Octave", sent_data->getInterval().toString().c_str() );
            _matlab->invoke( file );
        }
        else
        {
            TaskInfo("MatlabOperation::read(%s) Is waiting for Matlab/Octave to finish", I.toString().c_str() );
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

        OperationCache::invalidate_samples( Signal::Intervals::Intervals_ALL );
    }

    if (plotlines)
        plotlines->clear();
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

    if (_settings)
    {
        _settings->operation = this;
    }
}

} // namespace Adapters
