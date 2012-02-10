#include "compareimages.h"

#include "sawe/configuration.h"

#include <QtTest/QtTest>
#include <QGLWidget>
#include <QImage>
#include <QColor>
#include <QFile>

#ifdef min
#undef min
#undef max
#endif


CompareImages::
        CompareImages( QString testName )
:    limit(30)
{
    QString target;
    switch (Sawe::Configuration::computationDeviceType())
    {
    case Sawe::Configuration::DeviceType_Cuda: target = "cuda"; break;
    case Sawe::Configuration::DeviceType_OpenCL: target = "opencl"; break;
    case Sawe::Configuration::DeviceType_CPU: target = "cpu"; break;
    default: target = "unknown_target"; break;
    }

    resultFileName = QString("%1-%2-result.png").arg(testName).arg(target);
    goldFileName = QString("%1-%2-gold.png").arg(testName).arg(target);
    diffFileName = QString("%1-%2-diff.png").arg(testName).arg(target);

    QFile::remove(resultFileName);
}


void CompareImages::
        saveImage(Sawe::pProject p)
{
    saveImage(p->mainWindowWidget(), p->tools().render_view()->glwidget);
}


void CompareImages::
        saveImage(QWidget* mainwindow, QGLWidget *glwidget)
{
    TaskTimer ti("CompareImages::saveImage");

    glwidget->makeCurrent();

    QPixmap pixmap(mainwindow->size());
    QGL::setPreferredPaintEngine(QPaintEngine::OpenGL);
    QPainter painter(&pixmap);
    // Note that Microsoft Windows does not allow an application to interrupt
    // what the user is currently doing in another application.
    mainwindow->activateWindow();
    mainwindow->raise();
    mainwindow->render(&painter);
    QImage glimage = glwidget->grabFrameBuffer();

    QPoint p2 = glwidget->mapTo( mainwindow, QPoint() );
    painter.drawImage(p2, glimage);

    pixmap.save(resultFileName);
}


void CompareImages::
        verifyResult()
{
    if (!QFile::exists(goldFileName))
    {
        QFAIL( QString("Couldn't find expected image '%1' to compare against the "
                       "result image '%2'. If this is the first time you're running "
                       "this test you could create the expected image by renaming "
                       "'%2' to '%1'.")
               .arg(goldFileName)
               .arg(resultFileName)
               .toLocal8Bit().data());
    }

    QImage goldimage(goldFileName);
    QImage resultimage(resultFileName);

    QCOMPARE( goldimage.size(), resultimage.size() );

    QImage diffImage( goldimage.size(), goldimage.format() );

    double diff = 0;
    for (int y=0; y<goldimage.height(); ++y)
    {
        for (int x=0; x<goldimage.width(); ++x)
        {
            float gold = QColor(goldimage.pixel(x,y)).lightnessF();
            float result = QColor(resultimage.pixel(x,y)).lightnessF();
            diff += std::fabs( gold - result );
            float greenoffset = 1./3;
            float hue = fmod(10 + greenoffset + (gold - result)*0.5f, 1.f);
            diffImage.setPixel( x, y,
                                QColor::fromHsvF(
                                        hue,
                                        std::min(1.f, gold - result == 0
                                                      ? 0
                                                      : 0.5f+0.5f*std::fabs( gold - result )),
                                        0.5f+0.5f*gold
                                ).rgba() );
        }
    }

    diffImage.save( diffFileName );

    TaskInfo("compareImages, ligtness difference between '%s' and '%s' was %g, tolerated max difference is %g. Saved diff image in '%s'",
             goldFileName.toStdString().c_str(), resultFileName.toStdString().c_str(),
             diff, limit, diffFileName.toStdString().c_str() );

    QVERIFY(diff <= limit);
}
