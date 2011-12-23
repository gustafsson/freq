#include "sawe/project_header.h"
#include <redirectstdout.h>
#include <cuda_gl_interop.h>
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

void mappedVboTestCuda( DataStorage<float>::Ptr datap );

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
    pVbo vbo( new Vbo(1024, GL_ARRAY_BUFFER, GL_STATIC_DRAW));
    MappedVbo<float> mapping(vbo, DataStorageSize(256,1,1));
    DataStorage<float>::Ptr copy( new DataStorage<float>(*mapping.data) );
	mappedVboTestCuda( copy );
    QVERIFY2(true, "Failure");
}

QTEST_MAIN(MappedVboTest);

#include "tst_mappedvbotest.moc"
