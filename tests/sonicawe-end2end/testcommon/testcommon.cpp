#include "sawetest.h"
#include "compareimages.h"
#include "callslotevent.h"

#include "sawe/application.h"

#include <QtTest/QtTest>
#include <QGLWidget>

class TestCommon : public SaweTestClass
{
    Q_OBJECT
public:
    TestCommon();

    virtual void projectOpened();
    virtual void finishedWorkSection(int workSectionCounter);

private slots:
    void saweTestClassTest();
    void compareImagesTestResult();
    void compareImagesTest();
    void calledSlotTest();

protected slots:
    void hasCalledSlotTestSlot();

private:
    CompareImages compare;

    bool project_was_opened_;
    bool has_done_work_;
    bool has_called_slot_;
};


TestCommon::
        TestCommon()
            :
    project_was_opened_(false),
    has_done_work_(false),
    has_called_slot_(false)
{}


void TestCommon::
        saweTestClassTest()
{
    project( Sawe::Application::global_ptr()->slotNew_recording( -1 ) );

    exec();

    QVERIFY( project_was_opened_ );
    QVERIFY( has_done_work_ );
    QVERIFY( has_called_slot_ );
}


void TestCommon::
        projectOpened()
{
    project_was_opened_ = true;

    compare.saveImage( project() );

    QWidget* glwidget = project()->tools().render_view()->glwidget;

    QTestEventList tel;
    tel.push_back( new CallSlotEvent(this, SLOT(hasCalledSlotTestSlot())) );
    tel.simulate(glwidget);

    SaweTestClass::projectOpened();
}


void TestCommon::
        finishedWorkSection(int /*workSectionCounter*/)
{
    has_done_work_ = true;
}


void TestCommon::
        compareImagesTestResult()
{
    compare.verifyResult();
}


void TestCommon::
        compareImagesTest()
{
    CompareImages ci("predefined");
    QFile::copy(ci.goldFileName, ci.resultFileName);
    ci.verifyResult();
}


void TestCommon::
        calledSlotTest()
{
    QVERIFY( has_called_slot_ );
}


void TestCommon::
        hasCalledSlotTestSlot()
{
    has_called_slot_ = true;

    Sawe::Application::global_ptr()->slotClosed_window( project()->mainWindowWidget() );
}

SAWETEST_MAIN(TestCommon)

#include "testcommon.moc"
