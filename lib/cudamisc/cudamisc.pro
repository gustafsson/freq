# -------------------------------------------------
# Project created by QtCreator 2009-11-08T21:47:47
# -------------------------------------------------


####################

TARGET = cudamisc
TEMPLATE = lib
win32:TEMPLATE = vclib
win32:CONFIG += debug_and_release

CONFIG += staticlib warn_on
CONFIG += c++11 buildflags
CONFIG += tmpdir
CONFIG += precompile_header_with_all_headers

QT += opengl # to use QGLWidget
QT += widgets # to use QApplication

DEFINES += CUDAMISC_LIBRARY


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


OTHER_FILES += \
    *.pro \
