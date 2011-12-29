#ifndef COMPAREIMAGES_H
#define COMPAREIMAGES_H

#include <QString>

#include "sawe/project.h"

class QWidget;
class QGLWidget;

class CompareImages
{
public:
    CompareImages( QString testName = "test" );

    QString resultFileName, goldFileName, diffFileName;
    double limit;

    void saveImage(Sawe::pProject p);
    void verifyResult();

private:
    void saveImage(QWidget* mainwindow, QGLWidget *glwidget);
};


#endif // COMPAREIMAGES_H
