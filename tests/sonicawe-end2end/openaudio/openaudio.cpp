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
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;
using namespace Tfr;
using namespace Signal;

class OpenAudio : public SaweTestClass
{
    Q_OBJECT

public:
    OpenAudio();

private slots:
    void initOpenAudio();
    void openAudio();

    void verifyResult();

protected slots:
    void saveImage();

private:
    virtual void projectOpened();
    virtual void finishedWorkSection(int workSectionCounter);

    QString sourceAudio;

    CompareImages compareImages;
};


OpenAudio::
        OpenAudio()
{
    sourceAudio = "music-1.ogg";

#ifdef USE_CUDA
    compareImages.limit = 50.;
#else
    compareImages.limit = 100.;
#endif
}


void OpenAudio::
        initOpenAudio()
{
    project( Sawe::Application::global_ptr()->slotOpen_file( sourceAudio.toStdString() ) );
}


void OpenAudio::
        openAudio()
{
    exec();
}


void OpenAudio::
        projectOpened()
{
    SaweTestClass::projectOpened();
}


void OpenAudio::
        finishedWorkSection(int workSectionCounter)
{
    if (0!=workSectionCounter)
        return;

    QTimer::singleShot(1, this, SLOT(saveImage()));
}


void OpenAudio::
        saveImage()
{
    compareImages.saveImage( project() );

    Sawe::Application::global_ptr()->slotClosed_window( project()->mainWindowWidget() );
}


void OpenAudio::
        verifyResult()
{
    compareImages.verifyResult();
}


SAWETEST_MAIN(OpenAudio)

#include "openaudio.moc"
