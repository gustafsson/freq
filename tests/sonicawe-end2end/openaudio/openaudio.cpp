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

#include "sawe/application.h"
#include "sawe/project.h"
#include "tools/renderview.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;
using namespace Tfr;
using namespace Signal;

class OpenAudio : public QObject
{
    Q_OBJECT

public:
    OpenAudio();

private slots:
    void initTestCase();
    void cleanupTestCase();
    void openAudio();
    void compareImages();

protected slots:
    void setInitialized();
    void postSaveImage();
    void saveImage();

private:
    int n;
    Sawe::Project* p;
    QImage resultImage, goldImage;

    QString resultFileName, goldFileName, diffFileName;

    QString sourceAudio;

    bool initialized, hasPostedScreenshot;
};


OpenAudio::
        OpenAudio()
            :
            initialized(false),
            hasPostedScreenshot(false)
{
#ifdef USE_CUDA
    resultFileName = "opengui-result-cuda.png";
    goldFileName = "opengui-gold-cuda.png";
    diffFileName = "opengui-diff-cuda.png";
#else
    resultFileName = "opengui-result-cpu.png";
    goldFileName = "opengui-gold-cpu.png";
    diffFileName = "opengui-diff-cpu.png";
#endif

    sourceAudio = "music-1.ogg";
}


void OpenAudio::
        initTestCase()
{
    p = Sawe::Application::global_ptr()->slotOpen_file( sourceAudio.toStdString() ).get();
    connect( p->tools().render_view(), SIGNAL(postPaint()), SLOT(setInitialized()));
    connect( p->tools().render_view(), SIGNAL(finishedWorkSection()), SLOT(postSaveImage()));
    QFile::remove(resultFileName);
}


void OpenAudio::
        setInitialized()
{
    if (initialized)
        return;

    initialized = true;

    if (!p->mainWindow()->getItems()->actionToggleTimelineWindow->isChecked())
        p->mainWindow()->getItems()->actionToggleTimelineWindow->trigger();
}


void OpenAudio::
    postSaveImage()
{
    if (hasPostedScreenshot)
        return;

    hasPostedScreenshot = true;

    QTimer::singleShot(0, this, SLOT(saveImage()));
}


void OpenAudio::
        saveImage()
{
    TaskTimer ti("saveImage");

    QGLWidget* glwidget = p->tools().render_view()->glwidget;
    glwidget->makeCurrent();

    QPixmap pixmap(p->mainWindowWidget()->size());
    QGL::setPreferredPaintEngine(QPaintEngine::OpenGL);
    QPainter painter(&pixmap);
    p->mainWindowWidget()->render(&painter);
    QImage glimage = glwidget->grabFrameBuffer();

    QPoint p2 = glwidget->mapTo( p->mainWindowWidget(), QPoint() );
    painter.drawImage(p2, glimage);

    resultImage = pixmap.toImage();
    resultImage.save(resultFileName);

    Sawe::Application::global_ptr()->slotClosed_window( p->mainWindowWidget() );
}


void OpenAudio::
        compareImages()
{
    QImage openguigold(goldFileName);

    QCOMPARE( openguigold.size(), resultImage.size() );

    QImage diffImage( openguigold.size(), openguigold.format() );

    double diff = 0;
    for (int y=0; y<openguigold.height(); ++y)
    {
        for (int x=0; x<openguigold.width(); ++x)
        {
            float gold = QColor(openguigold.pixel(x,y)).lightnessF();
            float result = QColor(resultImage.pixel(x,y)).lightnessF();
            diff += std::fabs( gold - result );
            float hue = fmod(10 + (gold - result)*0.5f, 1.f);
            diffImage.setPixel( x, y, QColor::fromHsvF( hue, std::min(1.f, gold - result == 0 ? 0 : 0.5f+0.5f*std::fabs( gold - result )), 0.5f+0.5f*gold ).rgba() );
        }
    }

    diffImage.save( diffFileName );

    double limit = 50.;
    TaskInfo("OpenGui::compareImages, ligtness difference between '%s' and '%s' was %g, tolerated max difference is %g. Saved diff image in '%s'",
             goldFileName.toStdString().c_str(), resultFileName.toStdString().c_str(),
             diff, limit, diffFileName.toStdString().c_str() );

    QVERIFY(diff < limit);
}


void OpenAudio::
        cleanupTestCase()
{
}


void OpenAudio::
        openAudio()
{
    TaskTimer ti("openAudio");

    Sawe::Application::global_ptr()->exec();
}

// expanded QTEST_MAIN but for Sawe::Application
int main(int argc, char *argv[])
{
    std::vector<const char*> argvector(argc+2);
    for (int i=0; i<argc; ++i)
        argvector[i] = argv[i];

    argvector[argc++] = "--use_saved_gui_state=0";
    argvector[argc++] = "--skip_update_check=1";

    Sawe::Application application(argc, (char**)&argvector[0], false);
    QTEST_DISABLE_KEYPAD_NAVIGATION
    OpenAudio tc;
    return QTest::qExec(&tc, argc, (char**)&argvector[0]);
}

#include "openaudio.moc"
