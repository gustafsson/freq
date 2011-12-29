// gpumisc
#include "gl.h"
#include "datastorage.h"

// Qt
#include <QtCore/QString>
#include <QtTest/QtTest>
#include <QtCore/QCoreApplication>
#include <QGLWidget>


class MappedVboTestNoCopy : public QObject
{
    Q_OBJECT

public:
    MappedVboTestNoCopy();

    QGLWidget a;

private Q_SLOTS:
    void initTestCase();
    void cleanupTestCase();
    void testCase1();
};

MappedVboTestNoCopy::MappedVboTestNoCopy()
{
}

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "mappedvbo.h"
#include "cudaPitchedPtrType.h"
#include "CudaException.h"
#endif

void mappedVboTestCuda( DataStorage<float>::Ptr datap );

void MappedVboTestNoCopy::initTestCase()
{
    a.show(); // glew needs an OpenGL context
}

void MappedVboTestNoCopy::cleanupTestCase()
{
}

void MappedVboTestNoCopy::testCase1()
{
#ifdef USE_CUDA
    QVERIFY( cudaSuccess == cudaGLSetGLDevice( 0 ) );
#ifndef __APPLE__ // glewInit is not needed on Mac
    QVERIFY( 0==glewInit() );
#endif
    pVbo vbo( new Vbo(1024, GL_ARRAY_BUFFER, GL_STATIC_DRAW));
    MappedVbo<float> mapping(vbo, DataStorageSize(256,1,1));
    mappedVboTestCuda( mapping.data );
    QVERIFY2(true, "Failure");
#endif
}

QTEST_MAIN(MappedVboTestNoCopy);

#include "tst_mappedvbotestnocopy.moc"
