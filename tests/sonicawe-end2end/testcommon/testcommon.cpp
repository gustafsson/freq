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
    TaskTimer ti("%s::%s", vartype(*this).c_str(), __FUNCTION__, NULL);

    project( Sawe::Application::global_ptr()->slotNew_recording() );

    exec();

    QVERIFY( project_was_opened_ );
    QVERIFY( has_called_slot_ );
#ifdef __APPLE__
    QVERIFY( !has_done_work_ );
#else
    QVERIFY( has_done_work_ );
#endif
}


void TestCommon::
        projectOpened()
{
    TaskTimer ti("%s::%s", vartype(*this).c_str(), __FUNCTION__, NULL);

    SaweTestClass::projectOpened();

    project_was_opened_ = true;

    QWidget* glwidget = project()->tools().render_view()->glwidget;
    QTestEventList tel;
    tel.push_back( new CallSlotEvent(this, SLOT(hasCalledSlotTestSlot())) );
    tel.simulate(glwidget);

    compare.saveImage( project() );
}


void TestCommon::
        finishedWorkSection(int /*workSectionCounter*/)
{
    has_done_work_ = true;
}


void TestCommon::
        compareImagesTestResult()
{
    TaskTimer ti("%s::%s", vartype(*this).c_str(), __FUNCTION__, NULL);

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
    TaskTimer ti("%s::%s", vartype(*this).c_str(), __FUNCTION__, NULL);

    has_called_slot_ = true;

    Sawe::Application::global_ptr()->slotClosed_window( project()->mainWindowWidget() );
}

SAWETEST_MAIN(TestCommon)

#include "testcommon.moc"
