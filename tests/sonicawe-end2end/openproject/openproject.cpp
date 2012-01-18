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
#include <qtestcase.h>

#include "sawetest.h"
#include "compareimages.h"

#include "sawe/application.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;
using namespace Tfr;
using namespace Signal;

class OpenProject : public SaweTestClass
{
    Q_OBJECT

public:
    OpenProject();

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
    QString projectName;
    QString goldProjectName;

    Sawe::pProject projectAudio;
    Sawe::pProject projectProjectFile;

    CompareImages compareAudioImages;
    CompareImages compareProjectImages;
};


OpenProject::
        OpenProject()
            :
            compareAudioImages("audio"),
            compareProjectImages("project")
{
    sourceAudio = "music-1.ogg";
    projectName = "musicproject.sonicawe";
    goldProjectName = "musicprojectgold.sonicawe";

    compareProjectImages.goldFileName = compareAudioImages.goldFileName;

#ifdef USE_CUDA
    compareAudioImages.limit = 60.;
    compareProjectImages.limit = 100.;
#else
    compareAudioImages.limit = 60.;
    compareProjectImages.limit = 100.;
#endif
}


void OpenProject::
        initOpenAudio()
{
    project( projectAudio = Sawe::Application::global_ptr()->slotOpen_file( sourceAudio.toStdString() ) );

    Tools::RenderView* view = projectAudio->toolRepo().render_view();
    Tools::RenderModel* model = view->model;

    model->renderer->y_scale = 0.01f;
    model->_qx = 63.4565;
    model->_qy = 0;
    model->_qz = 0.37;
    model->_px = 0;
    model->_py = 0;
    model->_pz = -10;
    model->_rx = 46.2;
    model->_ry = 253.186;
    model->_rz = 0;

    model->orthoview.reset( model->_rx >= 90 );

    view->emitTransformChanged();
}


void OpenProject::
        openAudio()
{
    exec();
}


void OpenProject::
        projectOpened()
{
    SaweTestClass::projectOpened();
}


void OpenProject::
        finishedWorkSection(int /*workSectionCounter*/)
{
    QTimer::singleShot(1, this, SLOT(saveImage()));
}


void OpenProject::
        saveImage()
{
    static int saveImageCounter = 0;
    switch (saveImageCounter++)
    {
    case 0:
        {
            compareAudioImages.saveImage( projectAudio );
            for (int i=0; i<6; ++i)
            {
                if (i>=3 && i<=4)
                    QFile::remove(projectName);

                if (0 == i%3)
                    projectAudio->saveAs(projectName.toStdString());
                else
                    projectAudio->save();

                QByteArray savedProject = QFile(projectName).readAll();
                QByteArray goldProject = QFile(goldProjectName).readAll();

                if (   savedProject.size() != goldProject.size()
                    || memcmp( savedProject.data(), goldProject.data(), goldProject.size() ) != 0 )
                {
                    QFAIL(QString("Files %1 and %2 differ").arg(projectName).arg(goldProjectName).toLocal8Bit().data());
                }
            }

            project( projectProjectFile = Sawe::Application::global_ptr()->slotOpen_file( projectName.toStdString() ) );
            Sawe::Application::global_ptr()->slotClosed_window( projectAudio->mainWindowWidget() );
            projectAudio.reset();
            break;
        }
    case 1:
        {
            compareProjectImages.saveImage( projectProjectFile );
            Sawe::Application::global_ptr()->slotClosed_window( projectProjectFile->mainWindowWidget() );
            projectProjectFile.reset();
            break;
        }
    }

    return;
}


void OpenProject::
        verifyResult()
{
    compareAudioImages.verifyResult();
    compareProjectImages.verifyResult();
}


SAWETEST_MAIN(OpenProject)

#include "openproject.moc"
