#include <boost/noncopyable.hpp>
#include <boost/exception/all.hpp>

#include "squircle.h"
#include "chain.h"
#include "openurl.h"
#include "touchnavigation.h"
#include "selection.h"
#include "optimaltimefrequencyresolution.h"

#include "prettifysegfault.h"
#include "log.h"

#include <QGuiApplication>
#include <QQuickView>
#include <QQmlEngine>
#include <QQmlApplicationEngine>

/**
 * @brief The MyGuiApplication class should log uncaught exceptions during
 * event processing.
 */
class MyGuiApplication: public QGuiApplication
{
public:
    MyGuiApplication(int argc, char**argv)
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

int main(int argc, char *argv[])
{
    Log("Started app");
    for (int i=0; i<argc; i++)
        Log("argv[%d] = %s") % i % argv[i];

    PrettifySegfault::setup ();

    MyGuiApplication app(argc, argv);
    app.setOrganizationName("Frekk Consulting");\
    app.setOrganizationDomain("frekk.consulting");\
    app.setApplicationName(QFileInfo(app.applicationFilePath()).baseName());\
    app.setApplicationDisplayName ("Frekk");
    app.setApplicationName ("Frekk");

    qmlRegisterType<Squircle>("OpenGLUnderQML", 1, 0, "Squircle");
    qmlRegisterType<Chain>("OpenGLUnderQML", 1, 0, "Chain");
    qmlRegisterType<OpenUrl>("OpenGLUnderQML", 1, 0, "OpenUrl");
    qmlRegisterType<TouchNavigation>("OpenGLUnderQML", 1, 0, "TouchNavigation");
    qmlRegisterType<Selection>("OpenGLUnderQML", 1, 0, "Selection");
    qmlRegisterType<OptimalTimeFrequencyResolution>("OpenGLUnderQML", 1, 0, "OptimalTimeFrequencyResolution");

    int r = 1;
    QWindow* window;
    QQmlEngine* engine;

    QQuickView view;
    QQmlApplicationEngine appengine;
    QUrl qml {"qrc:/main.qml"};

    bool use_qquickview = false;
    if (use_qquickview)
    {
        // QQuickView doesn't create an OS X application window with an icon
        // for maximizing to fullscreen

        view.setResizeMode(QQuickView::SizeRootObjectToView);
        view.setSource(qml);
        window = &view;
        engine = view.engine ();
    }
    else
    {
        appengine.load (qml);

        QObject* root = appengine.rootObjects().count () > 0 ? appengine.rootObjects().at(0) : 0;
        window = dynamic_cast<QWindow*>(root);
        engine = &appengine;

        if (!window)
            Log("main: root element is not ApplicationWindow, use QQuickView instead");
    }

    if (window)
    {
        Log("main: window type %d") % window->type ();

        // http://qt-project.org/wiki/How_to_use_OpenGL_Core_Profile_with_Qt
        bool enableLegacyOpenGL = true;
        if (!enableLegacyOpenGL) {
            QSurfaceFormat f = window->format();
            f.setProfile(QSurfaceFormat::CoreProfile);
            f.setVersion(4, 4);
            window->setFormat(f);
        }

        QObject::connect(engine, SIGNAL(quit()), &app, SLOT(quit()));
        window->show();
        r = app.exec();
    }

    Log("Closing app");
    return r;
}
