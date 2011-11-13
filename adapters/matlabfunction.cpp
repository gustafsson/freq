#include "matlabfunction.h"

// gpumisc
#include "TaskTimer.h"

// qt
#include <QFileInfo>
#include <QTemporaryFile>
#include <QDir>
#include <QErrorMessage>
#include <QSettings>

// boost
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/algorithm/string.hpp>

// std
#include <sys/stat.h>
#include <fstream>

using namespace std;
using namespace boost;
using namespace boost::posix_time;


//#define TIME_MatlabFunction
#define TIME_MatlabFunction if(0)

namespace Adapters {


MatlabFunctionSettings& MatlabFunctionSettings::
        operator=(const MatlabFunctionSettings& b)
{
    arguments(b.arguments());
    chunksize(b.chunksize());
    computeInOrder(b.computeInOrder());
    operation = b.operation;
    redundant(b.redundant());
    scriptname(b.scriptname());
    argumentdescription(b.argumentdescription());

    return *this;
}


bool MatlabFunctionSettings::
        isTerminal()
{
    return scriptname().empty();
}


bool MatlabFunctionSettings::
        isSource()
{
    return chunksize()==-2;
}


void MatlabFunctionSettings::
        setAsSource()
{
    chunksize(-2);
}


DefaultMatlabFunctionSettings::
        DefaultMatlabFunctionSettings()
            :
            chunksize_(0),
            computeInOrder_(0),
            redundant_(0),
            argumentdescription_("Arguments")
{}


DefaultMatlabFunctionSettings::
        DefaultMatlabFunctionSettings(const MatlabFunctionSettings& b)
{
    *this = b;
}


DefaultMatlabFunctionSettings& DefaultMatlabFunctionSettings::
        operator=(const MatlabFunctionSettings& b)
{
    MatlabFunctionSettings::operator =(b);
    return *this;
}


void DefaultMatlabFunctionSettings::
        setProcess(QProcess*)
{
}


bool startProcess(QProcess* pid, const QString& name, const QStringList& args)
{
    pid->start(name, args);
    pid->waitForStarted();

    if (pid->state() == QProcess::Running)
    {
        TaskInfo ti("Started: %s", name.toStdString().c_str());
        foreach (QString a, args)
            TaskInfo("%s", a.toStdString().c_str());

        return true;
    }
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
        MatlabFunction( string f, float timeout, MatlabFunctionSettings* settings )
:   _pid(0),
    _hasCrashed(false),
    _timeout( timeout )
{
    _matlab_filename = QFileInfo(f.c_str()).fileName().toStdString();
    _matlab_function = QFileInfo(f.c_str()).baseName().toStdString();

    init(f, settings);
}


MatlabFunction::
        MatlabFunction( QString f, QString subname, float timeout, MatlabFunctionSettings* settings )
:   _pid(0),
    _hasCrashed(false),
    _timeout( timeout )
{
    _matlab_filename = QFileInfo(f).fileName().toStdString();
    _matlab_function = (QFileInfo(f).baseName() + "_" + subname).toStdString();

    init(f.toStdString(), settings);
}


void MatlabFunction::
        init(string fullpath, MatlabFunctionSettings* settings)
{
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
            QApplication::applicationDirPath().replace("\\", "\\\\").replace("'", "\\'" ) + "/matlab",
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

        if (fullpath.empty())
        {

        }
        else
        {
            string path = QFileInfo(fullpath.c_str()).path().replace("'", "\\'") .toStdString();
            string filename = QString(fullpath.c_str()).replace("'", "\\'") .toStdString();
            matlab_command
                    << "source('" << filename << "');"
                    << "addpath('" << path << "');"
                    << "sawe_filewatcher('" << _dataFile << "',@" << _matlab_function;
            octave_command
                    << "source('" << filename << "');"
                    << "addpath('" << path << "');"
                    << "sawe_filewatcher('" << _dataFile << "',@" << _matlab_function;

            string arguments = settings ? settings->arguments() : "";

            trim( arguments );
            trim_if( arguments, is_any_of(";") );

            if (arguments.size())
            {
                TaskInfo ti("arguments(%d) = %s", arguments.size(), arguments.c_str());
                for (unsigned i=0; i<arguments.size(); ++i)
                    TaskInfo("%d", arguments[i]);

                matlab_command << ", " << arguments.c_str();
                octave_command << ", " << arguments.c_str();
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

        if (fullpath.empty())
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
        _pid->setProcessChannelMode( QProcess::MergedChannels );
        if (settings) settings->setProcess( _pid );

        QString defaultscript = QSettings().value("defaultscript", "matlab").toString();
        QString octavepath = QSettings().value("octavepath", "").toString();
        QString matlabpath = QSettings().value("matlabpath", "").toString();

        for (int i=0; i<2; i++)
        {
            int j = (i + (defaultscript != "matlab"))%2;
            if (0 == j)
            {
                if (!fullpath.empty())
                {
                    QStringList matlab_names;
                    matlab_names.push_back(matlabpath);
                    matlab_names.push_back("matlab");
                    if (startProcess(_pid, matlab_names, matlab_args))
                        return;

                    TaskInfo("Couldn't start MATLAB");
                }
            }

            if (1 == j)
            {
                QStringList octave_names;
                octave_names.push_back(octavepath);
                octave_names.push_back("octave-3.2.3");
                octave_names.push_back("octave");
                if (startProcess(_pid, octave_names, octave_args))
                    return;

                TaskInfo("Couldn't start Octave");
            }
        }

#ifdef _WIN32
        if (!f.empty())
        {
            TaskInfo("Trying common installation paths for MATLAB instead");
            QStringList matlab_paths;
            matlab_paths.push_back("C:\\Program Files\\MATLAB\\R2008b\\bin\\matlab.exe");
            matlab_paths.push_back("C:\\Program Files (x86)\\MATLAB\\R2008b\\bin\\matlab.exe");

            if (startProcess(_pid, matlab_paths, matlab_args))
                return;

            TaskInfo("Couldn't start Matlab");
        }
#endif

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
        finished(int exitCode, QProcess::ExitStatus /*exitStatus*/)
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
        invoke( string source )
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


string MatlabFunction::
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


string MatlabFunction::
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
#elif defined(_WIN32)
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


QProcess* MatlabFunction::
        getProcess()
{
    return _pid;
}


string MatlabFunction::
        matlabFunction()
{
    return _matlab_function;
}


string MatlabFunction::
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
    throw invalid_argument("Timeout in MatlabFunction::invokeAndWait");
}

} // namespace Adapters
