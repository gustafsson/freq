#include "sawe/application.h"

#include "reader.h"

// Sonic AWE
#include "ui/mainwindow.h"
#include "tfr/cwt.h"

// gpumisc
#include <demangle.h>
#include <computationkernel.h>

// std
#include <sstream>

// qt
#include <QTime>
#include <QtGui/QMessageBox>
#include <QGLWidget>
#include <QSettings>
#include <QDesktopServices>

#ifdef USE_CUDA
// gpumisc
#include <gpucpudatacollection.h>

// cuda
#include "cuda.h"
#endif

using namespace std;

#define STRINGIFY(x) #x
#define TOSTR(x) STRINGIFY(x)

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

static void show_fatal_exception_qt( const string& str )
{
    QMessageBox::critical( 0,
                 QString("Error, closing application"),
				 QString::fromLocal8Bit(str.c_str()) );
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
    stringstream ss;
    ss   << "Error: " << vartype(x) << endl
         << "Message: " << x.what();
    return ss.str();
}

static string fatal_unknown_exception_string() {
    return "Error: An unknown error occurred";
}

Application::
        Application(int& argc, char **argv, bool dont_parse_sawe_argument )
:   QApplication(argc, argv),
    default_record_device(-1)
{
    shared_glwidget_ = new QGLWidget(QGLFormat(QGL::SampleBuffers));

    setOrganizationName("MuchDifferent");
    setOrganizationDomain("muchdifferent.com");

    #if !defined(TARGET_reader)
        setApplicationName("Sonic AWE");
    #else
        setApplicationName("Sonic AWE Reader");
    #endif


    if (!dont_parse_sawe_argument)
        parse_command_line_options(argc, argv); // will call 'exit(0)' on invalid arguments
}

Application::
        ~Application()
{
    TaskInfo ti("Closing Sonic AWE, %s", _version_string.c_str());
    ti.tt().partlyDone();

    _projects.clear();

    delete shared_glwidget_;
}

Application* Application::
        global_ptr() 
{
    Application* app = dynamic_cast<Application*>(QApplication::instance());
    BOOST_ASSERT( app );
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


string Application::
        version_string()
{
    global_ptr()->build_version_string();
    return global_ptr()->_version_string;
}


string Application::
        title_string()
{
    global_ptr()->build_version_string();
    return global_ptr()->_title_string;
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
    //} catch (const exception &x) {
    } catch (const invalid_argument &x) {
        const char* what = x.what();
        if (1 == QMessageBox::warning( 0,
                                       QString("Couldn't complete the requested action"),
                                       QString("Couldn't complete the requested action.\nDetails on the error follow:\n\n")+
                                       QString::fromLocal8Bit(what),
                                       "Ignore", "Exit program", QString::null, 0, 0 ))
        {
            err = fatal_exception_string(x);
        }
    } catch (const exception &x) {
        err = fatal_exception_string(x);
    } catch (...) {
        err = fatal_unknown_exception_string();
    }

    if (!err.empty() && _fatal_error.empty())
    {
        _fatal_error = err;
        show_fatal_exception( err );
        this->exit(-2);
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
        if (!q->isModified() && q->worker.number_of_samples() == 0)
            q->mainWindow()->close();
    }

    if ("not"==Reader::reader_text().substr(0,3))
        return;

    _projects.insert( p );

    apply_command_line_options( p );

    p->setModified( false );
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
#ifdef USE_CUDA
    GpuCpuDataCollection::moveAllDataToCpuMemory();
#endif

    TaskInfo("Reset CWT");
    Tfr::Cwt::Singleton().resetSingleton();


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
        slotNew_recording( int record_device )
{
    TaskTimer tt("New recording");
    if (record_device<0)
        record_device = QSettings().value("inputdevice", -1).toInt();

    pProject p = Project::createRecording( record_device );
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
        build_version_string()
{
    if (!_version_string.empty())
        return;

    stringstream ss;
    #ifdef SONICAWE_VERSION
        ss << "v" << TOSTR(SONICAWE_VERSION);
    #else
        ss << "dev " << __DATE__;
        #ifdef _DEBUG
            ss << ", " << __TIME__;
        #endif

        #ifdef SONICAWE_BRANCH
            if( 0 < strlen( TOSTR(SONICAWE_BRANCH) ))
                ss << " - branch: " << TOSTR(SONICAWE_BRANCH);
        #endif
    #endif

    _version_string = ss.str();
    _title_string = Reader::reader_title() + " - " + _version_string;
}


void Application::
        check_license()
{
    Reader::reader_text(true);

    global_ptr()->_version_string.clear();
    global_ptr()->build_version_string();
}


} // namespace Sawe
