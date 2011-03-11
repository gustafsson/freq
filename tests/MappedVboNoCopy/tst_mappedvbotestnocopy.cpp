#include "sawe/project_header.h"
#include <redirectstdout.h>
#include <QtCore/QString>
#include <QtTest/QtTest>
#include <QtCore/QCoreApplication>
#include <QGLWidget>

class MappedVboTestNoCopy : public QObject
{
    Q_OBJECT

public:
    MappedVboTestNoCopy();
    RedirectStdout rs;

    QGLWidget a;

private Q_SLOTS:
    void initTestCase();
    void cleanupTestCase();
    void testCase1();
};

MappedVboTestNoCopy::MappedVboTestNoCopy()
: rs(__FILE__ " log.txt")
{
}

#include "mappedvbo.h"
#include "cudaPitchedPtrType.h"
#include "CudaException.h"

void mappedVboTestCuda( cudaPitchedPtrType<float> data );

void MappedVboTestNoCopy::initTestCase()
{
    a.show(); // glew needs an OpenGL context
}

void MappedVboTestNoCopy::cleanupTestCase()
{
}

void MappedVboTestNoCopy::testCase1()
{
    QVERIFY( cudaSuccess == cudaGLSetGLDevice( 0 ) );
    QVERIFY( 0==glewInit() );
    pVbo vbo( new Vbo(1024));
    MappedVbo<float> mapping(vbo, make_cudaExtent(256,1,1));
    mappedVboTestCuda( mapping.data->getCudaGlobal() );
    QVERIFY2(true, "Failure");
}

QTEST_MAIN(MappedVboTestNoCopy);

#include "tst_mappedvbotestnocopy.moc"
