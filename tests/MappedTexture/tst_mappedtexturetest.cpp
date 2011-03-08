#if 0
#include "sawe/project_header.h"
#include <QtCore/QString>
#include <QtTest/QtTest>
#include <QtCore/QCoreApplication>
#include <QGLWidget>

class MappedVboTest : public QObject
{
    Q_OBJECT

public:
    MappedVboTest();

    QGLWidget a;

private Q_SLOTS:
    void initTestCase();
    void cleanupTestCase();
    void testCase1();

};

MappedVboTest::MappedVboTest()
{
}

#include "mappedvbo.h"
#include "cudaPitchedPtrType.h"
#include "CudaException.h"

void mappedVboTestCuda( cudaPitchedPtrType<float> data );

void MappedVboTest::initTestCase()
{
    a.show(); // glew needs an OpenGL context
}

void MappedVboTest::cleanupTestCase()
{
}

void MappedVboTest::testCase1()
{
    QVERIFY( cudaSuccess == cudaGLSetGLDevice( 0 ) );
    QVERIFY( 0==glewInit() );
    pVbo vbo( new Vbo(1024));
    MappedVbo<float> mapping(vbo, make_cudaExtent(256,1,1));
    mappedVboTestCuda( mapping.data->getCudaGlobal() );
    QVERIFY2(true, "Failure");
}

QTEST_MAIN(MappedVboTest);

#include "tst_mappedvbotest.moc"
#endif
