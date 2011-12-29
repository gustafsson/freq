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
            Ui::MainWindow* ui = main->getItems();
            QWidget* glwidget = project()->tools().render_view()->glwidget;

            ui->actionPeakSelection->trigger();

            QTestEventList tel;
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(636, 177), 100);
//            tel.addMousePress(Qt::LeftButton, 0, QPoint(636, 176), 100);
//            tel.addMouseMove(QPoint(636, 177), 100);
//            tel.addMouseRelease(Qt::LeftButton, 0, QPoint(636, 178), 100);
#ifndef _MSC_VER
    #ifdef USE_CUDA
            tel.addMouseMove(QPoint(940, 319), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(940, 319), 100);
    #else
            tel.addMouseMove(QPoint(621, 187), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(621, 187), 100);
    #endif
#else
    #ifdef USE_CUDA
            tel.addMouseMove(QPoint(661, 202), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(661, 202), 100);
    #else
            tel.addMouseMove(QPoint(650, 219), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(650, 219), 100);
    #endif
#endif
            tel.simulate(glwidget);

//            QTestMouseEvent(QTest::MousePress, Qt::LeftButton, 0, QPoint(940, 318), 100).simulate(glwidget);
//            QTestMouseEvent(QTest::MouseMove, Qt::LeftButton, 0, QPoint(940, 319), 100).simulate(glwidget);
//            QTestMouseEvent(QTest::MouseRelease, Qt::LeftButton, 0, QPoint(940, 320), 100).simulate(glwidget);
//            QTestMouseEvent(QTest::MouseClick, Qt::LeftButton, 0, QPoint(940, 319), 100).simulate(glwidget);

            ui->actionActionRemove_selection->trigger();

            QTestMouseEvent(QTest::MouseMove, Qt::NoButton, 0, QPoint(440, 150), 100).simulate(glwidget);
            QTestMouseEvent(QTest::MouseClick, Qt::LeftButton, 0, QPoint(440, 150), 100).simulate(glwidget);
            ui->actionActivateNavigation->trigger();

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
