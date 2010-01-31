# -------------------------------------------------
# Project created by QtCreator 2009-11-06T11:26:14
# -------------------------------------------------
QT += opengl \
    testlib
TARGET = sonicawe
TEMPLATE = app
SOURCES += main.cpp \
    mainwindow.cpp \
    displaywidget.cpp \
    spectrogram.cpp \
    transform-inverse.cpp \
    filter.cpp \
    transform-chunk.cpp \
    transform.cpp \
    waveform.cpp \
    spectrogram-vbo.cpp \
    spectrogram-renderer.cpp
HEADERS += mainwindow.h \
    displaywidget.h \
    waveform.h \
    spectrogram.h \
    transform-inverse.h \
    filter.h \
    transform-chunk.h \
    transform.h \
    wavelet.cu.h \
    spectrogram-vbo.h \
    spectrogram-renderer.h \
    spectrogram-slope.cu.h
FORMS += mainwindow.ui
OTHER_FILES += wavelet.cu \
    spectrogram.frag \
    spectrogram.vert \
    spectrogram-slope.cu
CUDA_SOURCES += wavelet.cu \
    spectrogram-slope.cu
unix:IS64 = $$system(if [ -n "`uname -m | grep x86_64`" ];then echo 64; fi)
INCLUDEPATH += ../misc
unix:INCLUDEPATH += /usr/local/cuda/include
unix:LIBS += -lsndfile \
    -laudiere \
    -L/usr/local/cuda/lib$$IS64 \
    -lcuda \
    -lcufft \
    -L../misc \
    -lmisc \
    -lGLEW \
    -lGLU \
    -lGL
win32:LIBS += 
MOC_DIR = tmp/
OBJECTS_DIR = tmp/
UI_DIR = tmp/

# #######################################################################
# CUDA
# #######################################################################
win32 { 
    INCLUDEPATH += $(CUDA_INC_DIR)
    QMAKE_LIBDIR += $(CUDA_LIB_DIR)
    LIBS += -lcudart
    cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = $(CUDA_BIN_DIR)/nvcc.exe \
        -c \
        -Xcompiler \
        $$join(QMAKE_CXXFLAGS,",") \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        ${QMAKE_FILE_NAME} \
        -o \
        ${QMAKE_FILE_OUT}
}
unix { 
    # auto-detect CUDA path
    # CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
    # manual
    CUDA_DIR = /usr/local/cuda
    INCLUDEPATH += $$CUDA_DIR/include
    QMAKE_LIBDIR += $$CUDA_DIR/lib$$IS64
    LIBS += -lcudart
    cuda.output = $${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $${CUDA_DIR}/bin/nvcc \
        -c \
        -Xcompiler \
        $$join(QMAKE_CXXFLAGS,",") \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        ${QMAKE_FILE_NAME} \
        -o \
        ${QMAKE_FILE_OUT}
    cuda.dependcy_type = TYPE_C
    cuda.depend_command = nvcc \
        -M \
        -Xcompiler \
        $$join(QMAKE_CXXFLAGS,",") \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        ${QMAKE_FILE_NAME} \
        | \
        sed \
        "s,^.*: ,," \
        | \
        sed \
        "s,^ *,," \
        | \
        tr \
        -d \
        '\\\n'
}
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_UNIX_COMPILERS += cuda
