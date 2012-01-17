#include "sawe/project_header.h"
#include <redirectstdout.h>
#include <QtCore/QString>
#include <QtTest/QtTest>
#include <QtCore/QCoreApplication>
#include <QGLWidget>

class NetworkAccessTest : public QObject
{
    Q_OBJECT

public:
    NetworkAccessTest();

    QGLWidget a;

private Q_SLOTS:
    void initTestCase();
    void cleanupTestCase();
    void testCase1();
};

NetworkAccessTest::NetworkAccessTest()
{
}

#ifdef USE_CUDA
#include <cuda_gl_interop.h>

#include "mappedvbo.h"
#endif

void mappedVboTestCuda( DataStorage<float>::Ptr datap );

void NetworkAccessTest::initTestCase()
{
    a.show(); // glew needs an OpenGL context
}

void NetworkAccessTest::cleanupTestCase()
{
}

void NetworkAccessTest::testCase1()
{
#ifdef USE_CUDA
    QVERIFY( cudaSuccess == cudaGLSetGLDevice( 0 ) );
#ifndef __APPLE__ // glewInit is not needed on Mac
    QVERIFY( 0==glewInit() );
#endif
    pVbo vbo( new Vbo(1024, GL_ARRAY_BUFFER, GL_STATIC_DRAW));
    MappedVbo<float> mapping(vbo, DataStorageSize(256,1,1));
    DataStorage<float>::Ptr copy( new DataStorage<float>(*mapping.data) );
	mappedVboTestCuda( copy );
    QVERIFY2(true, "Failure");
#endif
}

QTEST_MAIN(NetworkAccessTest);

#include "tst_mappedvbotest.moc"
