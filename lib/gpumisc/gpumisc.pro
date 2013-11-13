# -------------------------------------------------
# Project created by QtCreator 2009-11-08T21:47:47
# -------------------------------------------------


####################

TARGET = gpumisc
TEMPLATE = lib
win32:TEMPLATE = vclib
win32:CONFIG += debug_and_release
#QT = core # gpumisc uses QMutex from Qt library in TaskTimer

CONFIG += staticlib warn_on
CONFIG += c++11 buildflags
QT += opengl

#DEFINES += NO_TASKTIMER_MUTEX
#DEFINES += THREADCHECKER_NO_CHECK
#DEFINES += CUDA_MEMCHECK_TEST
DEFINES += GPUMISC_LIBRARY

SOURCES += \
    backtrace.cpp \
    cpumemorystorage.cpp \
    cpuproperties.cpp \
    datastorage.cpp \
    debugbuf.cpp \
    detectgdb.cpp \
    demangle.cpp \
    exceptionassert.cpp \
    geometricalgebra.cpp \
    GlException.cpp \
    glframebuffer.cpp \
    glinfo.cpp \
    glprojection.cpp \
    GlTexture.cpp \
    gluunproject.cpp \
    mappedvbovoid.cpp \
    neat_math.cpp \
    prettifysegfault.cpp \
    redirectstdout.cpp \
    signalname.cpp \
    stringprintf.cpp \
    TaskTimer.cpp \
    ThreadChecker.cpp \
    timer.cpp \
    vbo.cpp \
    volatileptr.cpp \

HEADERS += \
    backtrace.h \
    computationkernel.h \
    cpumemoryaccess.h \
    cpumemorystorage.h \
    datastorage.h \
    datastorageaccess.h \
    cpuproperties.h \
    cva_list.h \
    debugbuf.h \
    debugmacros.h \
    debugstreams.h \
    demangle.h \
    detectgdb.h \
    deprecated.h \
    exceptionassert.h \
    expectexception.h \
    gl.h \
    GlException.h \
    glframebuffer.h \
    glinfo.h \
    glprojection.h \
    glPushContext.h \
    GlTexture.h \
    GLvector.h \
    geometricalgebra.h \
    gluunproject.h \
    gpumisc_global.h \
    HasSingleton.h \
    InvokeOnDestruction.hpp \
    mappedvbo.h \
    mappedvbovoid.h \
    msc_stdc.h \
    neat_math.h \
    operate.h \
    prettifysegfault.h \
    redirectstdout.h \
    redirectStream.h \
    releaseaftercontext.h \
    resample.h \
    resamplecpu.h \
    resamplehelpers.h \
    resampletypes.h \
    signalname.h \
    Statistics.h \
    StatisticsRandom.h \
    stringprintf.h \
    TAni.h \
    TaskTimer.h \
    texturereader.cu.h \
    ThreadChecker.h \
    throwInvalidArgument.h \
    timer.h \
    tmatrix.h \
    tmatrixstring.h \
    tvector.h \
    tvectorstring.h \
    unsignedf.h \
    unused.h \
    vbo.h \
    volatileptr.h \

win32 {
    SOURCES += StackWalker.cpp
    HEADERS += StackWalker.h
}

useopencl {
DEFINES += USE_OPENCL

SOURCES += \
    openclcontext.cpp \
    openclexception.cpp \
    openclmemorystorage.cpp \

HEADERS += \
    openclcontext.h \
    openclexception.h \
    openclmemoryaccess.h \
    openclmemorystorage.h \

    gpuamd:INCLUDEPATH += "$(AMDAPPSDKROOT)include"
} #useopencl


usecuda {
DEFINES += USE_CUDA

SOURCES += \
    CudaException.cpp \
    cudaglobalstorage.cpp \
    cudaMemcpy3Dfix.cpp \
    CudaProperties.cpp \
    cudaUtil.cpp \
    cuffthandlecontext.cpp \

HEADERS += \
    CudaException.h \
    cudaglobalaccess.h \
    cudaglobalstorage.h \
    cudaKernels.h \
    cudaMemcpy3Dfix.h \
    cudaPitchedPtrType.h \
    CudaProperties.h \
    cudatemplates.cu.h \
    cudaUtil.h \
    cuda_vector_types_op.h \
    cuffthandlecontext.h \
    resamplecuda.cu.h \
    texturereader.cu.h \

!win32:INCLUDEPATH += /usr/local/cuda/include
win32:INCLUDEPATH += "\$(CUDA_INC_PATH)"

} #usecuda


win32: INCLUDEPATH += \
    ../sonicawe-winlib \
    ../sonicawe-winlib/glew/include

macx:INCLUDEPATH += \
    /opt/local/include/


####################
# Temporary output

usecuda {
  TMPDIR = tmp/cuda
} else:useopencl {
  TMPDIR = tmp/opencl
} else {
  TMPDIR = tmp/cpu
}

OBJECTS_DIR = $${TMPDIR}/

CONFIG(debug, debug|release):OBJECTS_DIR = $${OBJECTS_DIR}debug/
else:OBJECTS_DIR = $${OBJECTS_DIR}release/


OTHER_FILES += \
    LICENSE \
    *.pro \

win32 { 
    othersources.input = OTHER_FILES
    othersources.output = ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_COMPILERS += othersources
}
