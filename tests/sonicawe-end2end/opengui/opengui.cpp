#include "sawe/project_header.h"
#include <QtCore/QString>
#include <QtTest/QtTest>
#include <QtCore/QCoreApplication>
#include <iostream>
#include <QGLWidget> // libsonicawe uses gl, so we need to include a gl header in this project as well
#include <QTimer>
#include <QImage>
#include <QPainter>
#include <QRgb>

#include "sawetest.h"
#include "compareimages.h"

#include "sawe/application.h"
#include "sawe/project.h"
#include "tools/renderview.h"

using namespace std;
using namespace Tfr;
using namespace Signal;

class OpenGui : public SaweTestClass
{
    Q_OBJECT

public:
    OpenGui();

private slots:
    void initOpenGui();
    void openGui();
    void verifyResult();

protected slots:
    void saveImage();

private:
    virtual void projectOpened();
    virtual void finishedWorkSection(int workSectionCounter);

    CompareImages compareImages;
};


OpenGui::
        OpenGui()
{
}


void OpenGui::
        initOpenGui()
{
    project( Sawe::Application::global_ptr()->slotNew_recording( ) );
}


void OpenGui::
        openGui()
{
    exec();
}


void OpenGui::
        projectOpened()
{
    SaweTestClass::projectOpened();

    QTimer::singleShot(1, this, SLOT(saveImage()));
}


void OpenGui::
        finishedWorkSection(int workSectionCounter)
{
    if (0!=workSectionCounter)
        return;
}


void OpenGui::
        saveImage()
{
    compareImages.saveImage( project() );

    Sawe::Application::global_ptr()->slotClosed_window( project()->mainWindowWidget() );
}


void OpenGui::
        verifyResult()
{
    compareImages.verifyResult();
}


SAWETEST_MAIN(OpenGui)

#include "opengui.moc"
