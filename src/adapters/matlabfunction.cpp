#include "matlabfunction.h"

// gpumisc
#include "tasktimer.h"

// qt
#include <QtCore> // QSettings, QDir, QTemporaryFile, QFileInfo
#include <QtWidgets> // QApplication, QErrorMessage
#include <QtOpenGL> // QGLWidget

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
    overlap(b.overlap());
    scriptname(b.scriptname());
    argument_description(b.argument_description());

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


void MatlabFunctionSettings::
        print(const char*str)
{
    TaskInfo ti("%s", str);
    TaskInfo("chunksize = %d", chunksize());
    TaskInfo("computeInOrder = %s", computeInOrder()?"true":"false");
    TaskInfo("overlap = %d", overlap());
    TaskInfo("scriptname = %s", scriptname().c_str());
    TaskInfo("arguments = %s", arguments().c_str());
    TaskInfo("argument_description = %s", argument_description().c_str());
    TaskInfo("operation = %p", operation);
    TaskInfo("isTerminal = %s", isTerminal()?"true":"false");
    TaskInfo("isSource = %s", isSource()?"true":"false");
}



DefaultMatlabFunctionSettings::
        DefaultMatlabFunctionSettings()
            :
            chunksize_(0),
            computeInOrder_(0),
            redundant_(0),
            argument_description_("Arguments")
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
        MatlabFunction( QString f, QString subname, float timeout, MatlabFunctionSettings* settings, bool justtest )
:   _pid(0),
    _hasCrashed(false),
    _timeout( timeout )
{
    _matlab_filename = QFileInfo(f).fileName().toStdString();
    if (!subname.isEmpty())
        subname = "_" + subname;
    _matlab_function = (QFileInfo(f).baseName() + subname).toStdString();

    init(f.toStdString(), settings, justtest, false);
}


void MatlabFunction::
        init(string fullpath, MatlabFunctionSettings* settings, bool justtest, bool sendoutput)
{
    { // Set filenames
        QTemporaryFile tempFile(QDir::tempPath() + QDir::separator() + "saweinterop.XXXXXX");
        tempFile.setAutoRemove(false);
        tempFile.open(); // create file and block other instances from picking the same name
        _interopName = tempFile.fileName();

        TaskInfo("MatlabFunction: Reserved %s for %s", _interopName.toStdString().c_str(), _matlab_function.c_str());
        _dataFile = _interopName.toStdString() + ".h5";
        _resultFile = _dataFile + ".result.h5";

        if (!sendoutput)
            _dataFile = _resultFile;
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
        else if(justtest)
        {
            string path = QFileInfo(fullpath.c_str()).path().replace("'", "\\'") .toStdString();
            string filename = QString(fullpath.c_str()).replace("'", "\\'") .toStdString();
            matlab_command
                    << "addpath('" << path << "');"
                    << "f=@" << _matlab_function << ";";
            octave_command
                    << "source('" << filename << "');"
                    << "addpath('" << path << "');"
                    << "f=@" << _matlab_function << ";";

            matlab_command << ");";
            octave_command << ");";
        }
        else
        {
            string path = QFileInfo(fullpath.c_str()).path().replace("'", "\\'") .toStdString();
            string filename = QString(fullpath.c_str()).replace("'", "\\'") .toStdString();
            matlab_command
                    << "try;"
                    << "addpath('" << path << "');"
                    << "f=@" << _matlab_function << ";"
                    << "catch;exit;end;"
                    << "sawe_filewatcher('" << _dataFile << "',f";
            octave_command
                    << "try;"
                    << "source('" << filename << "');"
                    << "addpath('" << path << "');"
                    << "f=@" << _matlab_function << ";"
                    << "catch;exit;end;"
                    << "sawe_filewatcher('" << _dataFile << "',f";

            string arguments = settings ? settings->arguments() : "";

            if (arguments.size() )
            {
                matlab_command << ", {" << arguments.c_str() << "}";
                octave_command << ", {" << arguments.c_str() << "}";
            }

            matlab_command << ");";
            octave_command << ");";
        }

        QStringList matlab_args;
        //matlab_args.push_back("-noFigureWindows");
        //matlab_args.push_back("-nojvm");
        matlab_args.push_back("-nodesktop");
        matlab_args.push_back("-nosplash");
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

        bool tryMatlab = _matlab_function == QFileInfo(_matlab_filename.c_str()).baseName().toStdString();

        for (int i=0; i<2; i++)
        {
            int j = (i + (defaultscript != "matlab"))%2;
            if (0 == j && tryMatlab)
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
        if (!fullpath.empty() && tryMatlab)
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

    TaskInfo("MatlabFunction: Releasing %s from %s", _interopName.toStdString().c_str(), _matlab_function.c_str());

    QFile::remove(_interopName);
    QFile::remove(_dataFile.c_str());
    QFile::remove(getTempName().c_str());
    QFile::remove(_resultFile.c_str());
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
        _hasCrashed = true;
    }
    else
    {
        _hasCrashed = false;
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
        std::this_thread::sleep_for (std::chrono::milliseconds(10));
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
