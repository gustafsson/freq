#include <boost/noncopyable.hpp>
#include <boost/exception/all.hpp>

#include "squircle.h"
#include "chain.h"
#include "openurl.h"
#include "touchnavigation.h"
#include "selection.h"
#include "optimaltimefrequencyresolution.h"
#include "showprocessing.h"

#include "prettifysegfault.h"
#include "log.h"

#include <QGuiApplication>
#include <QQuickView>
#include <QQmlEngine>
#include <QQmlApplicationEngine>

#ifdef Q_OS_IOS
#define USE_QUICKVIEW
#endif

/**
 * @brief The MyGuiApplication class should log uncaught exceptions during
 * event processing.
 */
class MyGuiApplication: public QGuiApplication
{
public:
    MyGuiApplication(int& argc, char**argv)
        :
          QGuiApplication(argc,argv)
    {}

    bool notify(QObject *o, QEvent *e) override
    {
        try {
            return QGuiApplication::notify (o,e);
        } catch (...) {
            Log("main.cpp: notify(%s,%d)\n%s")
                    % o->objectName ().toStdString () % int(e->type ())
                    % boost::current_exception_diagnostic_information ();
            quit ();
            return false;
        }
    }
};


int run();

int main(int argc, char *argv[])
{
    for (int i=0; i<argc; i++)
        Log("argv[%d] = %s") % i % argv[i];

    PrettifySegfault::setup ();

    MyGuiApplication app(argc, argv);
    app.setOrganizationName("Freq Consulting");
    app.setOrganizationDomain("freq.consulting");
    app.setApplicationName(QFileInfo(app.applicationFilePath()).baseName());
    app.setApplicationDisplayName ("Freq");

    qmlRegisterType<Squircle>("OpenGLUnderQML", 1, 0, "Squircle");
    qmlRegisterType<Chain>("OpenGLUnderQML", 1, 0, "Chain");
    qmlRegisterType<OpenUrl>("OpenGLUnderQML", 1, 0, "OpenUrl");
    qmlRegisterType<TouchNavigation>("OpenGLUnderQML", 1, 0, "TouchNavigation");
    qmlRegisterType<Selection>("OpenGLUnderQML", 1, 0, "Selection");
    qmlRegisterType<OptimalTimeFrequencyResolution>("OpenGLUnderQML", 1, 0, "OptimalTimeFrequencyResolution");
    qmlRegisterType<ShowProcessing>("OpenGLUnderQML", 1, 0, "ShowProcessing");

    int r = run();
    app.processEvents ();
    QThread::currentThread ()->eventDispatcher ()->processEvents (QEventLoop::AllEvents);
    return r;
}


int run()
{
    int r = 1;
    QWindow* window;
    QQmlEngine* engine;

#ifdef USE_QUICKVIEW
    QQuickView view;
    // QQuickView doesn't create an OS X application window with an icon
    // for maximizing to fullscreen

    view.setResizeMode(QQuickView::SizeRootObjectToView);
    view.setSource(QUrl("qrc:///Main.qml"));
    window = &view;
    engine = view.engine ();
#else
    QQmlApplicationEngine appengine;
    appengine.load (QUrl("qrc:///app.qml"));

    QObject* root = appengine.rootObjects().count () > 0 ? appengine.rootObjects().at(0) : 0;
    window = dynamic_cast<QWindow*>(root);
    engine = &appengine;

    if (!window)
        Log("main: root element is not ApplicationWindow, use QQuickView instead");
#endif

    if (window)
    {
        QSurfaceFormat f = window->format();
#if !defined(LEGACY_OPENGL) && !defined(GL_ES_VERSION_2_0)
        // http://qt-project.org/wiki/How_to_use_OpenGL_Core_Profile_with_Qt
        // OS X has either a modern OpenGL (3.2+) with core profile or legacy OpenGL 2.1.
        // There is no compatibility profile and no other versions. Which version of modern
        // OpenGL you get depends on your hardware and version of OS X. I got 4.1 on 10.10 for instance.
        f.setProfile(QSurfaceFormat::CoreProfile);
        f.setVersion(3, 2);
#endif
        f.setSamples(4);
        window->setFormat(f);

        QObject::connect(engine, SIGNAL(quit()), QCoreApplication::instance (), SLOT(quit()));
        window->show();
        r = QCoreApplication::instance ()->exec ();
    }

    Log("Closing app");
    return r;
}
