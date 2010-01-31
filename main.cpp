#include <QtGui/QApplication>
#include <QtGui/QFileDialog>
#include "mainwindow.h"
#include "displaywidget.h"
#include "wavelettransform.h"
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    QString fileName = QFileDialog::getOpenFileName(0, "Open sound file");
  
    const char* soundfile = "file0255_2_in.wav";
    boost::shared_ptr<WavelettTransform> wt( new WavelettTransform(fileName.toAscii().constData()) );
    boost::shared_ptr<DisplayWidget> dw( new DisplayWidget( wt ) );

    w.setCentralWidget( dw.get() );
    dw->show();
    w.show();

    {
        cout << "blaj1" <<endl;
        //boost::shared_ptr<TransformData> blaj = wt->getWavelettTransform();
        cout << "blaj2" <<endl;
    }
        cout << "blaj3" <<endl;
    //dw.reset();
        cout << "blaj4" <<endl;
        //wt.reset();
        cout << "blaj5" <<endl;
   return a.exec();
   // return 0;
}
