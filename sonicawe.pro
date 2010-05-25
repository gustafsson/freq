# -------------------------------------------------
# Project created by QtCreator 2009-11-06T11:26:14
# -------------------------------------------------

### Compiler settings
TARGET = sonicawe
TEMPLATE = app
win32:TEMPLATE = vcapp
macx:CONFIG -= app_bundle

CONFIG += warn_on
QT += opengl
unix:QMAKE_CXXFLAGS_DEBUG += -ggdb
unix:QMAKE_CXXFLAGS_RELEASE += -O3
macx:QMAKE_CXXFLAGS_RELEASE += -O3

### Settings for using llvm instead of gcc on linux
#unix {
#    QMAKE_CXX = llvm-g++
#    QMAKE_CC = llvm-gcc
#    QMAKE_LINK = llvm-g++
#}

### Source code
RESOURCES += icon-resources.qrc
SOURCES += main.cpp \
    mainwindow.cpp \
    displaywidget.cpp \
    heightmap-glblock.cpp \
    selection.cpp \
    signal-source.cpp \
    signal-audiofile.cpp \
    signal-microphonerecorder.cpp \
    signal-operation.cpp \
    signal-operation-basic.cpp \
    signal-operation-composite.cpp \
    signal-samplesintervaldescriptor.cpp \
    signal-sink.cpp \
    signal-sinksource.cpp \
    signal-playback.cpp \
    layer.cpp \
    tfr-stft.cpp \
    tfr-cwt.cpp \
    tfr-filter.cpp \
    tfr-inversecwt.cpp \
    tfr-chunk.cpp \
    sawe-csv.cpp \
    signal-filteroperation.cpp \
    signal-worker.cpp \
    signal-writewav.cpp \
    heightmap-renderer.cpp \
    heightmap-collection.cpp \
    heightmap-reference.cpp
HEADERS += mainwindow.h \
    displaywidget.h \
    tfr-wavelet.cu.h \
    heightmap-glblock.h \
    heightmap-slope.cu.h \
    heightmap-block.cu.h \
    tfr-filter.cu.h \
    selection.h \
    heightmap-position.h \
    signal-source.h \
    signal-audiofile.h \
    signal-microphonerecorder.h \
    signal-operation.h \
    signal-operation-basic.h \
    signal-operation-composite.h \
    signal-samplesintervaldescriptor.h \
    signal-playback.h \
    signal-sink.h \
    signal-sinksource.h \
    layer.h \
    tfr-stft.h \
    tfr-cwt.h \
    tfr-filter.h \
    tfr-inversecwt.h \
    tfr-chunk.h \
    sawe-csv.h \
    sawe-mainplayback.h \
    signal-filteroperation.h \
    signal-worker.h \
    signal-writewav.h \
    heightmap-renderer.h \
    heightmap-collection.h \
    heightmap-reference.h
FORMS += mainwindow.ui
OTHER_FILES += tfr-wavelet.cu \
    heightmap.frag \
    heightmap.vert \
    heightmap-slope.cu \
    heightmap-block.cu \
    tfr-filter.cu
CUDA_SOURCES += tfr-wavelet.cu \
    heightmap-slope.cu \
    heightmap-block.cu \
    tfr-filter.cu
OTHER_SOURCES += heightmap.frag \
    heightmap.vert \
    sonicawe.pro
win32 { 
    othersources.input = OTHER_SOURCES
    othersources.output = ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_UNIX_COMPILERS += othersources
}
unix:IS64 = $$system(if [ -n "`uname -m | grep x86_64`" ];then echo 64; fi)
INCLUDEPATH += ../gpumisc
unix:DEFINES += SONICAWE_BRANCH="\'$$system(if [ -f .git/HEAD ];then cat .git/HEAD | sed -E "s/ref:\ refs\\\/heads\\\/master// | sed -E "s/ref:\ refs\\\/heads\\\///"; fi)\'"
unix:INCLUDEPATH += /usr/local/cuda/include
unix:LIBS = -lsndfile \
    -lGLEW \
    -lGLU \
    -lGL \
    -lglut \
    -lportaudiocpp -lportaudio
macx:INCLUDEPATH += /usr/local/cuda/include \
    ../../libs/include
macx:LIBS = -lsndfile \
    -L/usr/local/cuda/lib \
    -L../misc \
    -lmisc \
    -framework GLUT \
    -framework OpenGL \
    -L../../libs -lportaudiocpp -lportaudio
win32:INCLUDEPATH += \
	..\..\winlib\glut \
	..\..\winlib\glew\include \
	..\..\winlib\portaudio\include \
	..\..\winlib\libsndfile\include \
	..\..\winlib
win32:LIBS += \
	-l..\..\winlib\glut\glut32 \
	-l..\..\winlib\glew\lib\glew32 \
    -l..\..\winlib\libsndfile\libsndfile-1 \
	-l..\..\winlib\portaudio\portaudio \
	-l..\..\winlib\portaudio\portaudiocpp
LIBS += -lcufft
unix:LIBS += -L../gpumisc -lgpumisc
macx:LIBS += -L../gpumisc -lgpumisc

MOC_DIR = tmp
OBJECTS_DIR = tmp/
UI_DIR = tmp

# #######################################################################
# CUDA
# #######################################################################
win32 { 
    INCLUDEPATH += $(CUDA_INC_PATH)
    LIBS += -L$(CUDA_LIB_PATH) -lcudart
	QMAKE_CXXFLAGS -= -Zc:wchar_t-
	QMAKE_CXXFLAGS += -Zc:wchar_t
    cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = $(CUDA_BIN_PATH)/nvcc.exe \
        -c \
        -Xcompiler \
        \"$$join(QMAKE_CXXFLAGS," ")\" \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        --use_fast_math \
        ${QMAKE_FILE_BASE}.cu \
        -o \
        ${QMAKE_FILE_OUT}
}
unix {
    # auto-detect CUDA path
    # CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
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
        --use_fast_math \
        ${QMAKE_FILE_NAME} \
        -o \
        ${QMAKE_FILE_OUT}
    cuda.dependcy_type = TYPE_C
    cuda.depend_command_dosntwork = nvcc \
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

# cuda.depends = nvcc -M -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} | sed "s,^.*: ,," | sed "s,^ *,," | tr -d '\\\n'

macx { 
    # auto-detect CUDA path
    # CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
    # manual
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
        --use_fast_math \
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
# end of cuda section #######################################################################
