#include <QtGui/QApplication>
#include "mainwindow.h"
#include "displaywidget.h"
#include "wavelettransform.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    const char* soundfile = "file0255_2_in.wav";
    boost::shared_ptr<WavelettTransform> wt( new WavelettTransform(soundfile) );
    boost::shared_ptr<DisplayWidget> dw( new DisplayWidget( wt ) );

    w.setCentralWidget( dw.get() );
    dw->show();
    w.show();

    return a.exec();
}
