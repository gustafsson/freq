#include <QtCore/QString>
#include <QtTest/QtTest>
#include <QtCore/QCoreApplication>

class BlockTest : public QObject
{
    Q_OBJECT

public:
    BlockTest();

private Q_SLOTS:
    void initTestCase();
    void cleanupTestCase();
    void testCase1();
};

BlockTest::BlockTest()
{
}

void BlockTest::initTestCase()
{

}

void BlockTest::cleanupTestCase()
{
}

void BlockTest::testCase1()
{
    QVERIFY2(true, "Failure");
}

QTEST_MAIN(BlockTest);

#include "tst_blocktest.moc"
