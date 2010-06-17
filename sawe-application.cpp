#include "sawe-application.h"
#include <QTime>
#include <sstream>
#include <QtGui/QMessageBox>
#include <demangle.h>

using namespace std;

#define STRINGIFY(x) #x
#define TOSTR(x) STRINGIFY(x)

namespace Sawe {

// static members
Application*    Application::_app = 0;
std::string     Application::_fatal_error;

static void show_fatal_exception_cerr( const std::string& str )
{
    cerr << endl << endl
         << "======================" << endl
         << str << endl
         << "======================" << endl;
    cerr.flush();
}

static void show_fatal_exception_qt( const std::string& str )
{
    QMessageBox::critical( 0,
                 QString("Fatal error. Sonic AWE needs to close"),
                 QString::fromStdString(str) );
}

static void show_fatal_exception( const std::string& str )
{
    show_fatal_exception_cerr(str);
    show_fatal_exception_qt(str);
}

static string fatal_exception_string( const std::exception &x )
{
    std::stringstream ss;
    ss   << "Error: " << demangle(typeid(x).name()) << endl
         << "Message: " << x.what();
    return ss.str();
}

static string fatal_unknown_exception_string() {
    return "Error: An unknown error occurred";
}

Application::
        Application(int& argc, char **argv)
:   QApplication(argc, argv)
{
    BOOST_ASSERT( !_app );

    _app = this;
    _version_string = "Sonic AWE - development snapshot\n";

    QDateTime now = QDateTime::currentDateTime();
    now.date().year();
    stringstream ss;
    ss << "Sonic AWE";
#ifndef SONICAWE_RELEASE
    ss << " - ";
#ifdef SONICAWE_VERSION
    ss << TOSTR(SONICAWE_VERSION);
#else
    ss << __DATE__;// << " - " << __TIME__;
#endif
#endif

#ifdef SONICAWE_BRANCH
    if( 0 < strlen( TOSTR(SONICAWE_BRANCH) ))
        ss << " - branch: " << TOSTR(SONICAWE_BRANCH);
#endif

    _version_string = ss.str();
}

Application::
        ~Application()
{
    BOOST_ASSERT( _app );
    _app = 0;
}

Application* Application::
        global_ptr() {
    BOOST_ASSERT( _app );
    return _app;
}


void Application::
        display_fatal_exception()
{
    show_fatal_exception(fatal_unknown_exception_string());
}


void Application::
        display_fatal_exception(const std::exception& x)
{
    show_fatal_exception(fatal_exception_string(x));
}


bool Application::
        notify(QObject * receiver, QEvent * e)
{
    bool v = false;
    string err;

    try {
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
        this->exit(-2);
    }

    return v;
}

void Application::
		openadd_project( pProject p )
{
    setActiveWindow( 0 );
    setActiveWindow( p->mainWindow().get() );
    p->mainWindow()->activateWindow();
    _projects.push_back( p );
}

pProject Application::
        slotNew_recording( int record_device )
{
    TaskTimer tt("New recording1");
    pProject p = Project::createRecording( record_device );
    if (p)
		openadd_project(p);
    return p;
}

pProject Application::
        slotOpen_file( std::string project_file_or_audio_file )
{
    pProject p = Project::open( project_file_or_audio_file );
    if (p)
		openadd_project(p);
    return p;
}

void Application::
    slotClosed_window( QWidget* w )
{
    for (std::list<pProject>::iterator i = _projects.begin(); i!=_projects.end();)
    {
        if (w == dynamic_cast<QWidget*>((*i)->mainWindow().get()))
            i = _projects.erase( i );
        else
            i++;
    }
}

} // namespace Sawe
