#include <boost/noncopyable.hpp>

#include "squircle.h"
#include "prettifysegfault.h"
#include "log.h"


#include <QGuiApplication>
#include <QQuickView>

int main(int argc, char *argv[])
{
    Log("Enter main");

    PrettifySegfault::setup ();

    QGuiApplication app(argc, argv);

    qmlRegisterType<Squircle>("OpenGLUnderQML", 1, 0, "Squircle");

    QQuickView view;
    view.setResizeMode(QQuickView::SizeRootObjectToView);
//    view.setSource(QUrl("qrc:///scenegraph/openglunderqml/main.qml"));
    view.setSource(QUrl("qrc:/main.qml"));
    view.show();

    return app.exec();
}
