#include "sawe/application.h"

#include "reader.h"

// Sonic AWE
#include "ui/mainwindow.h"
#include "tfr/cwt.h"
#include "configuration.h"

// gpumisc
#include "demangle.h"
#include "computationkernel.h"
#include "glinfo.h"

// std
#include <sstream>

// qt
#include <QTime>
#include <QtGui/QMessageBox>
#include <QGLWidget>
#include <QSettings>
#include <QDesktopServices>
#include <QMouseEvent>
#include <QHostInfo>

// boost
#include <boost/date_time/posix_time/posix_time.hpp>


#ifdef USE_CUDA

// cuda
#include "cuda.h"
#endif


using namespace std;

namespace Sawe {

// static members
string     Application::_fatal_error;

static void show_fatal_exception_cerr( const string& str )
{
    cerr << endl << endl
         << "======================" << endl
         << str << endl
         << "======================" << endl;
    cerr.flush();
}

static void show_fatal_exception_qt( const string& /*str*/ )
{
    int t = time(0);
    srand(t);
    int r = rand();
    QString alt[] = {
        "Darn",
        "Oups",
        "Not again",
        "Uhm",
        "Ah well"
    };

    QString a = alt[r%(sizeof(alt)/sizeof(alt[0]))];
    QString name = QApplication::instance ()->applicationName ();
    QMessageBox::critical( 0,
                 a,
                 QString("%1, need to restart %2.\nThe log file may contain some cryptic details").arg (a).arg (name));
}

void Application::
        show_fatal_exception( const string& str )
{
    show_fatal_exception_cerr(str);
    if (QApplication::instance())
        show_fatal_exception_qt(str);
}

static string fatal_exception_string( const exception &x )
{
    return boost::diagnostic_information(x);
}

static string fatal_unknown_exception_string() {
    return "Error: An unknown error occurred";
}

Application::
        Application(int& argc, char **argv, bool prevent_log_system_and_execute_args )
:   QApplication(argc, argv),
    default_record_device(-1)
{
    QGLFormat glformat;
    shared_glwidget_ = new QGLWidget(glformat);
    shared_glwidget_->makeCurrent();

    setOrganizationName("MuchDifferent");
    setOrganizationDomain("muchdifferent.com");

    #if defined(TARGET_reader)
        setApplicationName("Sonic AWE Reader");
    #elif defined(TARGET_hast)
        setApplicationName("Sonic AWE LOFAR");
    #else
        setApplicationName("Sonic AWE");
    #endif

    if (!prevent_log_system_and_execute_args)
        logSystemInfo(argc, argv);

    Sawe::Configuration::resetDefaultSettings();
    Sawe::Configuration::parseCommandLineOptions(argc, argv);

    if (!Sawe::Configuration::use_saved_state())
    {
        QVariant value = QSettings().value("value");
        setApplicationName(applicationName() + " temp");
        QSettings().clear();
        QSettings().setValue("value", value);
    }

    if (!prevent_log_system_and_execute_args)
    {
        execute_command_line_options(); // will call 'exit(0)' on invalid arguments
    }
}

Application::
        ~Application()
{
    TaskInfo ti("Closing Sonic AWE, %s", Sawe::Configuration::version_string().c_str());
    ti.tt().partlyDone();

    _projects.clear();

    delete shared_glwidget_;
}


void Application::
     logSystemInfo(int& argc, char **argv)
{
    TaskInfo ti("Version: %s", Sawe::Configuration::version_string().c_str());
    boost::gregorian::date today = boost::gregorian::day_clock::local_day();
    boost::gregorian::date_facet* facet(new boost::gregorian::date_facet("%A %B %d, %Y"));
    ti.tt().getStream().imbue(std::locale(std::cout.getloc(), facet));
    ti.tt().getStream() << " started on " << today << endl;

    TaskInfo("Build timestamp for %s: %s, %s. Revision %s",
        Sawe::Configuration::uname().c_str(),
        Sawe::Configuration::build_date().c_str(), Sawe::Configuration::build_time().c_str(),
        Sawe::Configuration::revision().c_str());

    {
        TaskInfo ti2("%u command line argument%s", argc, argc==1?"":"s");
        for (int i=0; i<argc; ++i)
            TaskInfo("%s", argv[i]);
    }

    TaskInfo("Organization: %s", organizationName().toStdString().c_str());
    TaskInfo("Organization domain: %s", organizationDomain().toStdString().c_str());
    TaskInfo("Application name: %s", applicationName().toStdString().c_str());
    TaskInfo("OS: %s", Sawe::Configuration::operatingSystemName().c_str());
    TaskInfo("domain: %s", QHostInfo::localDomainName().toStdString().c_str());
    TaskInfo("hostname: %s", QHostInfo::localHostName().toStdString().c_str());
    TaskInfo("number of CPU cores: %d", Sawe::Configuration::cpuCores());
    TaskInfo("OpenGL information\n%s", glinfo::driver_info().c_str ());
}


Application* Application::
        global_ptr() 
{
    Application* app = dynamic_cast<Application*>(QApplication::instance());
    EXCEPTION_ASSERT( app );
    return app;
}


QString Application::
        log_directory()
{
    QString localAppDir = QDesktopServices::storageLocation(QDesktopServices::DataLocation);
    return localAppDir;
}


QGLWidget* Application::
        shared_glwidget()
{
    return global_ptr()->shared_glwidget_;
}


void Application::
        display_fatal_exception()
{
    show_fatal_exception(fatal_unknown_exception_string());
}


void Application::
        display_fatal_exception(const exception& x)
{
    show_fatal_exception(fatal_exception_string(x));
}


bool Application::
        notify(QObject * receiver, QEvent * e)
{
    bool v = false;
    string err;

    try {
        if (e)
        {
            QEvent::Type t = e->type();
            switch (t)
            {
                case QEvent::MouseButtonPress:
                {
                    QMouseEvent* m = static_cast<QMouseEvent*>(e);
                    TaskInfo("QEvent::MouseButtonPress button=%d at %d, %d (%d, %d) on %s %s %p",
                             m->button(), m->x(), m->y(), m->globalX(), m->globalY(), vartype(*receiver).c_str(), receiver->objectName().toStdString().c_str(), receiver);
                    break;
                }
                case QEvent::MouseButtonRelease:
                {
                    QMouseEvent* m = static_cast<QMouseEvent*>(e);
                    TaskInfo("QEvent::MouseButtonRelease button=%d at %d, %d (%d, %d) from %s %s %p",
                             m->button(), m->pos().x(), m->pos().y(), m->globalX(), m->globalY(), vartype(*receiver).c_str(), receiver->objectName().toStdString().c_str(), receiver);
                    break;
                }
                case QEvent::KeyPress:
                {
                    QKeyEvent* m = static_cast<QKeyEvent*>(e);
                    TaskInfo("QEvent::KeyPress key=%d on %s %s %p", m->key(), vartype(*receiver).c_str(), receiver->objectName().toStdString().c_str(), receiver);
                    break;
                }
                default:
                    break;
            }

            switch (t)
            {
                case QEvent::MouseButtonPress:
                case QEvent::KeyPress:
                case QEvent::Show:
                case QEvent::Enter:
                    foreach (pProject p, _projects)
                    {
                        if (receiver == p->mainWindow())
                            p->tools().render_view()->userinput_update( true, false );
                    }
                    break;

                default:
                    break;
            }
        }

        v = QApplication::notify(receiver,e);
    } catch (const exception &x) {
        err = fatal_exception_string(x);
    } catch (...) {
        err = fatal_unknown_exception_string();
    }

    if (!err.empty() && _fatal_error.empty())
    {
        _fatal_error = err;
        show_fatal_exception( err );
        this->exit(2);
    }

    return v;
}

void Application::
		openadd_project( pProject p )
{
    setActiveWindow( 0 );
    setActiveWindow( p->mainWindow() );

    if (1 == _projects.size())
    {
        pProject q = *_projects.begin();
        if (!q->isModified () && q->extent ().interval.get ().count() == 0)
            q->mainWindow()->close();
    }

    if ("not"==Reader::reader_text().substr(0,3))
        return;

    _projects.insert( p );

    apply_command_line_options( p );

    p->setModified( false );
}

bool Application::
        has_other_projects_than(Project*p)
{
    for (set<pProject>::iterator i = _projects.begin(); i!=_projects.end(); ++i)
    {
        if (&**i != p)
            return true;
    }
    return false;
}

void Application::
        clearCaches()
{
    TaskTimer tt("Application::clearCaches");
#ifdef USE_CUDA
    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);
    TaskInfo("Total Cuda memory: %g MB (of which %g MB is available)",
             total/1024.f/1024, free/1024.f/1024);
#endif
    emit clearCachesSignal();

    if ( !QGLContext::currentContext() ) // See RenderView::~RenderView()
        return;


#ifdef USE_CUDA
    TaskInfo("cudaThreadExit()");
    cudaThreadExit();

    int count;
    cudaError_t e = cudaGetDeviceCount(&count);
    TaskInfo("Number of CUDA devices=%u, error=%s", count, cudaGetErrorString(e));
    // e = cudaThreadExit();
    // tt.info("cudaThreadExit, error=%s", cudaGetErrorString(e));
    //CudaProperties::printInfo(CudaProperties::getCudaDeviceProp());
    //e = cudaSetDevice( 1 );
    //tt.info("cudaSetDevice( 1 ), error=%s", cudaGetErrorString(e));
    //e = cudaSetDevice( 0 );
    //tt.info("cudaSetDevice( 0 ), error=%s", cudaGetErrorString(e));
    void *p=0;
    e = cudaMalloc( &p, 10 );
    TaskInfo("cudaMalloc( 10 ), p=%p, error=%s", p, cudaGetErrorString(e));
    e = cudaFree( p );
    TaskInfo("cudaFree, error=%s", cudaGetErrorString(e));

    cudaMemGetInfo(&free, &total);
    TaskInfo("Total Cuda memory: %g MB (of which %g MB is available)",
             total/1024.f/1024, free/1024.f/1024);

    ComputationSynchronize();

    cudaGetLastError();
#endif
    glGetError();
}

pProject Application::
        slotNew_recording()
{
    pProject p = Project::createRecording();
    if (p)
        openadd_project(p);

    return p;
}

pProject Application::
         slotOpen_file( string project_file_or_audio_file )
{
    pProject p = Project::open( project_file_or_audio_file );
    if (p)
        openadd_project(p);

    return p;
}

void Application::
    slotClosed_window( QWidget* w )
{
    for (set<pProject>::iterator i = _projects.begin(); i!=_projects.end();)
    {
        if (w == dynamic_cast<QWidget*>((*i)->mainWindow()))
        {
            _projects.erase( i );
            i = _projects.begin();
        }
        else
            i++;
    }
}


void Application::
        check_license()
{
    Reader::reader_text(true);

    Sawe::Configuration::rebuild_version_string();

    emit global_ptr()->titleChanged();
}


} // namespace Sawe
