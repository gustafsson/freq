# -------------------------------------------------
# Project created by QtCreator 2009-11-06T11:26:14
# -------------------------------------------------
macx:CONFIG -= app_bundle
QT += opengl \
    testlib
TARGET = sonicawe
TEMPLATE = app
SOURCES += main.cpp \
    mainwindow.cpp \
    displaywidget.cpp \
    wavelettransform.cpp \
    transformdata.cpp \
    waveform.cpp
HEADERS += mainwindow.h \
    displaywidget.h \
    wavelettransform.h \
    transformdata.h \
    waveform.h
FORMS += mainwindow.ui
OTHER_FILES += wavelet.cu
CUDA_SOURCES += wavelet.cu
unix:IS64 = $$system(if [ -n "`uname -m | grep x86_64`" ];then echo 64; fi)
INCLUDEPATH += ../misc
unix:INCLUDEPATH += /usr/local/cuda/include
unix:LIBS += \
    -lsndfile \
    -laudiere \
    -L/usr/local/cuda/lib$$IS64 \
    -lcuda \
    -lcufft \
    -L../misc \
    -lmisc
macx:INCLUDEPATH += /usr/local/cuda/include
macx:LIBS += \
    -lsndfile \
    -laudiere \
    -L/usr/local/cuda/lib \
    -lcuda \
    -lcufft \
    -L../misc \
    -lmisc
win32:LIBS += \
    -l..\..\audiere\lib\audiere \
    -l..\..\libsndfile\libsndfile-1 \
    -LC:\CUDA\lib \
    -lcuda \
    -lcufft \
    -L../misc \
    -lmisc
MOC_DIR = tmp
OBJECTS_DIR = tmp/
UI_DIR = tmp

# #######################################################################
# CUDA
# #######################################################################
win32 { 
    INCLUDEPATH += $(CUDA_INC_PATH)\
	..\..\libsndfile\include \
	..\..\audiere\include \
	.
    QMAKE_LIBDIR += $(CUDA_LIB_PATH)
    LIBS += -lcudart
#    cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
#    cuda.commands = $(CUDA_BIN_DIR)/nvcc.exe \
#        -c \
#        -Xcompiler \
#        $$join(QMAKE_CXXFLAGS,",") \
#        $$join(INCLUDEPATH,'" -I "','-I "','"') \
#        ${QMAKE_FILE_NAME} \
#        -o \
#        ${QMAKE_FILE_OUT}
}
unix { 
    # auto-detect CUDA path
    #CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
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
#    cuda.depends = nvcc -M -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} | sed "s,^.*: ,," | sed "s,^ *,," | tr -d '\\\n'
}
macx { 
    # auto-detect CUDA path
    #CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
    CUDA_DIR = /usr/local/cuda
    INCLUDEPATH += $$CUDA_DIR/include
    QMAKE_LIBDIR += $$CUDA_DIR/lib
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
macx {
	cuda.input = CUDA_SOURCES
	QMAKE_EXTRA_UNIX_COMPILERS += cuda
}
unix {
	cuda.input = CUDA_SOURCES
	QMAKE_EXTRA_UNIX_COMPILERS += cuda
}
########################################################################
