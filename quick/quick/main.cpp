#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include "squircle.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    qmlRegisterType<Squircle>("OpenGLUnderQML", 1, 0, "Squircle");

    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:///main.qml")));

    return app.exec();
}
