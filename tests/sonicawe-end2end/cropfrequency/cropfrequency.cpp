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

class CropFrequency : public SaweTestClass
{
    Q_OBJECT

public:
    CropFrequency();

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


CropFrequency::
        CropFrequency()
{
    sourceAudio = "music-1.ogg";
}


void CropFrequency::
        initOpenAudio()
{
    project( Sawe::Application::global_ptr()->slotOpen_file( sourceAudio.toStdString() ) );
}


void CropFrequency::
        openAudio()
{
    exec();
}


void CropFrequency::
        projectOpened()
{
    TaskTimer tt("DeleteSelection::projectOpened");

    Tools::RenderController* rc = project()->tools().getObject<Tools::RenderController>();
    QVERIFY( 0!=rc );

    for (int i=0; i<40; ++i)
        rc->yscaleDecrease();

    SaweTestClass::projectOpened();
}

static void dragMouseEvent(QWidget *widget, Qt::MouseButton button,
                       Qt::KeyboardModifiers stateKey, QPoint pos, int delay=-1)
{
    QTEST_ASSERT(widget);
    if(delay > 0)
        QTest::qWait(delay);

    if (pos.isNull())
        pos = widget->rect().center();

    QTEST_ASSERT(button == Qt::NoButton || button & Qt::MouseButtonMask);
    QTEST_ASSERT(stateKey == 0 || stateKey & Qt::KeyboardModifierMask);

    stateKey &= static_cast<unsigned int>(Qt::KeyboardModifierMask);

    QMouseEvent me(QEvent::MouseMove, pos,  widget->mapToGlobal(pos), button, button, stateKey);
    QSpontaneKeyEvent::setSpontaneous(&me);
    if (!qApp->notify(widget, &me)) {
        QString warning = QString::fromLatin1("Mouse event \"%1\" not accepted by receiving widget");
        QTest::qWarn(warning.arg(QString::fromLatin1("MouseDrag")).toAscii().data());
    }
}

class DragMouseEvent: public QTestEvent
{
public:
    inline DragMouseEvent(Qt::MouseButton button,
            Qt::KeyboardModifiers modifiers, QPoint position, int delay)
        : _button(button), _modifiers(modifiers), _pos(position), _delay(delay) {}
    inline QTestEvent *clone() const { return new DragMouseEvent(*this); }

    inline void simulate(QWidget *w)
    {
        dragMouseEvent(w, _button, _modifiers, _pos, _delay);
    }

private:
    Qt::MouseButton _button;
    Qt::KeyboardModifiers _modifiers;
    QPoint _pos;
    int _delay;
};


void CropFrequency::
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

            ui->actionFrequencySelection->trigger();

            QTestEventList tel;
            tel.addMousePress(Qt::LeftButton, 0, QPoint(661, 174), 100);
            for (int y=174; y<=204; y++)
            {
                tel.append(new DragMouseEvent(Qt::LeftButton, 0, QPoint(661, y), 10));
            }
            tel.addMouseRelease(Qt::LeftButton, 0, QPoint(661, 234), 100);

            {
                TaskInfo ti("Simulating events");
                tel.simulate(glwidget);
            }

            {
                TaskInfo ti("Triggering actionActionRemove_selection");

                ui->actionCropSelection->trigger();
            }

            break;
        }

    case 1:
        QTimer::singleShot(1, this, SLOT(saveImage()));
        break;
    }
}


void CropFrequency::
        saveImage()
{
    compareImages.saveImage( project() );

    Sawe::Application::global_ptr()->slotClosed_window( project()->mainWindowWidget() );
}


void CropFrequency::
        verifyResult()
{
    compareImages.verifyResult();
}


void CropFrequency::
        cleanupVerifyResult()
{
}


SAWETEST_MAIN(CropFrequency)

#include "cropfrequency.moc"
