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

#include "tools/selections/peakcontroller.h"

using namespace std;
using namespace Tfr;
using namespace Signal;

class DeleteSelection : public SaweTestClass
{
    Q_OBJECT

public:
    DeleteSelection();

private slots:
    void initOpenAudio();
    void openAudio();

    void verifyResult();
    void cleanupVerifyResult();

protected slots:
    void saveImage();

private:
    virtual void projectOpened();
    virtual void finishedWorkSection(int workSectionCounter);

    QString sourceAudio;

    CompareImages compareImages;
};


DeleteSelection::
        DeleteSelection()
{
    sourceAudio = "music-1.ogg";
}


void DeleteSelection::
        initOpenAudio()
{
    project( Sawe::Application::global_ptr()->slotOpen_file( sourceAudio.toStdString() ) );
}


void DeleteSelection::
        openAudio()
{
    exec();
}


void DeleteSelection::
        projectOpened()
{
    TaskTimer tt("DeleteSelection::projectOpened");

    Tools::RenderController* rc = project()->tools().getObject<Tools::RenderController>();
    QVERIFY( 0!=rc );

    for (int i=0; i<40; ++i)
        rc->yscaleDecrease();

    SaweTestClass::projectOpened();
}


void DeleteSelection::
        finishedWorkSection(int finishedWorkSections)
{
    switch (finishedWorkSections)
    {
    case 0:
        {
            Ui::SaweMainWindow* main = project()->mainWindow();
            main->activateWindow();
            main->raise();

            Ui::MainWindow* ui = main->getItems();
            QWidget* glwidget = project()->tools().render_view()->glwidget;

            ui->actionPeakSelection->trigger();

            QTestEventList tel;
#ifdef _MSC_VER
    #ifdef USE_CUDA
            tel.addMouseMove(QPoint(661, 202), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(661, 202), 100);
    #else
            tel.addMouseMove(QPoint(650, 219), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(650, 219), 100);
    #endif
#elif defined(__APPLE__)
    #ifdef USE_CUDA
            tel.addMouseMove(QPoint(657, 205), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(657-2, 205-3), 100);
    #else
            tel.addMouseMove(QPoint(604, 237), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(650, 219), 100);
    #endif
#else
    #ifdef USE_CUDA
            tel.addMouseMove(QPoint(671, 214), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(671, 214), 100);
    #else
            tel.addMouseMove(QPoint(661, 234), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(661, 234), 100);
    #endif
#endif
            tel.simulate(glwidget);

            ui->actionActionRemove_selection->trigger();

            break;
        }

    case 1:
        QTimer::singleShot(1, this, SLOT(saveImage()));
        break;
    }
}


void DeleteSelection::
        saveImage()
{
    compareImages.saveImage( project() );

    Sawe::Application::global_ptr()->slotClosed_window( project()->mainWindowWidget() );
}


void DeleteSelection::
        verifyResult()
{
    compareImages.verifyResult();
}


void DeleteSelection::
        cleanupVerifyResult()
{
}


SAWETEST_MAIN(DeleteSelection)

#include "deleteselection.moc"
