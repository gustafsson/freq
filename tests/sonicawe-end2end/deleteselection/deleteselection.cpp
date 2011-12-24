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
#include "tools/selections/peakcontroller.h"
#include "tools/dropnotifyform.h"

using namespace std;
using namespace Tfr;
using namespace Signal;

class DeleteSelection : public QObject
{
    Q_OBJECT

public:
    DeleteSelection();

private slots:
    void initTestCase();
    void cleanupTestCase();
    void openAudio();
    void compareImages();

protected slots:
    void setInitialized();
    void finishedWorkSection();
    void saveImage();

private:
    int n;
    Sawe::Project* p;
    QImage resultImage, goldImage;

    QString resultFileName, goldFileName, diffFileName;

    QString sourceAudio;

    bool initialized;
    int finishedWorkSections;
    bool success;
};


DeleteSelection::
        DeleteSelection()
            :
            initialized(false),
            finishedWorkSections(0),
            success(false)
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


void DeleteSelection::
        initTestCase()
{
    p = Sawe::Application::global_ptr()->slotOpen_file( sourceAudio.toStdString() ).get();
    connect( p->tools().render_view(), SIGNAL(postPaint()), SLOT(setInitialized()));
    connect( p->tools().render_view(), SIGNAL(finishedWorkSection()), SLOT(finishedWorkSection()));
    QFile::remove(resultFileName);
}


void DeleteSelection::
        setInitialized()
{
    if (initialized)
        return;

    initialized = true;

    if (!p->mainWindow()->getItems()->actionToggleTimelineWindow->isChecked())
        p->mainWindow()->getItems()->actionToggleTimelineWindow->trigger();

    foreach (QObject* o, p->mainWindow()->centralWidget()->children())
    {
        if (Tools::DropNotifyForm* dnf = dynamic_cast<Tools::DropNotifyForm*>(o))
        {
            dnf->close();
        }
    }

    Tools::RenderController* rc = p->tools().getObject<Tools::RenderController>();
    QVERIFY( 0!=rc );

    for (int i=0; i<40; ++i)
        rc->yscaleDecrease();
}


class CallSlotEvent : public QTestEvent
{
public:
    CallSlotEvent(QObject* receiver, const char* slotname) : receiver(receiver), slotname(slotname) {}

    virtual void simulate(QWidget *)
    {
        QTimer::singleShot(1, receiver, slotname);
    }

    virtual QTestEvent *clone() const
    {
        return new CallSlotEvent(receiver, slotname);
    }

private:
    QObject* receiver;
    const char* slotname;
};


void DeleteSelection::
    finishedWorkSection()
{
    switch (finishedWorkSections++)
    {
    case 0:
        {
            Ui::SaweMainWindow* main = p->mainWindow();
            Ui::MainWindow* ui = main->getItems();
            QWidget* glwidget = p->tools().render_view()->glwidget;

            ui->actionPeakSelection->trigger();

            QTestEventList tel;
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(636, 177), 100);
//            tel.addMousePress(Qt::LeftButton, 0, QPoint(636, 176), 100);
//            tel.addMouseMove(QPoint(636, 177), 100);
//            tel.addMouseRelease(Qt::LeftButton, 0, QPoint(636, 178), 100);
#ifdef USE_CUDA
            tel.addMouseMove(QPoint(940, 319), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(940, 319), 100);
#else
            tel.addMouseMove(QPoint(621, 187), 100);
            tel.addMouseClick(Qt::LeftButton, 0, QPoint(621, 187), 100);
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

    success = true;
}


void DeleteSelection::
        compareImages()
{
    success = false;

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

    double limit = 30.;
    TaskInfo("OpenGui::compareImages, ligtness difference between '%s' and '%s' was %g, tolerated max difference is %g. Saved diff image in '%s'",
             goldFileName.toStdString().c_str(), resultFileName.toStdString().c_str(),
             diff, limit, diffFileName.toStdString().c_str() );

    QVERIFY(diff < limit);

    success = true;
}


void DeleteSelection::
        cleanupTestCase()
{
    QVERIFY( success );
}


void DeleteSelection::
        openAudio()
{
    TaskTimer ti("openAudio");

    Sawe::Application::global_ptr()->exec();

    QVERIFY( success );
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
    DeleteSelection tc;
    return QTest::qExec(&tc, argc, (char**)&argvector[0]);
}

#include "deleteselection.moc"
