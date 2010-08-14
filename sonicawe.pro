# -------------------------------------------------
# Project created by QtCreator 2009-11-06T11:26:14
# -------------------------------------------------


####################
# Compiler settings

TARGET = sonicawe
TEMPLATE = app
win32:TEMPLATE = vcapp
win32:CONFIG += debug_and_release
macx:CONFIG -= app_bundle

CONFIG += warn_on
QT += opengl
unix:QMAKE_CXXFLAGS_DEBUG += -ggdb
!win32:QMAKE_CXXFLAGS_RELEASE -= -O2
!win32:QMAKE_CXXFLAGS_RELEASE += -O3
win32:QMAKE_LFLAGS += /FORCE:MULTIPLE
QMAKE_CXXFLAGS_DEBUG += -D_DEBUG

#!macx&!win32: QMAKE_CXX = colorgcc
#macx:QMAKE_CXX = g++ # Should not need this macx: with CONFIG(unix|!macx) above

### Settings for using llvm instead of gcc on linux
llvm {
    QMAKE_CXX = llvm-g++
    QMAKE_CC = llvm-gcc
    QMAKE_LINK = llvm-g++
}


####################
# Source code

RESOURCES += icon-resources.qrc
SOURCES += main.cpp \
    mainwindow.cpp \
    displaywidget.cpp \
    heightmap-glblock.cpp \
    selection.cpp \
    signal-source.cpp \
    signal-audiofile.cpp \
    signal-operationcache.cpp \
    signal-microphonerecorder.cpp \
    signal-operation.cpp \
    signal-operation-basic.cpp \
    signal-operation-composite.cpp \
    signal-samplesintervaldescriptor.cpp \
    signal-sink.cpp \
    signal-sinksource.cpp \
    signal-postsink.cpp \
    signal-playback.cpp \
    layer.cpp \
    tfr-stft.cpp \
    tfr-cwt.cpp \
    tfr-filter.cpp \
    tfr-inversecwt.cpp \
    tfr-chunk.cpp \
    tfr-chunksink.cpp \
    sawe-application.cpp \
    sawe-csv.cpp \
    sawe-hdf5.cpp \
    sawe-matlaboperation.cpp \
    sawe-matlabfilter.cpp \
    sawe-project.cpp \
    sawe-timelinewidget.cpp \
    signal-filteroperation.cpp \
    signal-worker.cpp \
    signal-writewav.cpp \
    heightmap-renderer.cpp \
    heightmap-collection.cpp \
    heightmap-reference.cpp \
    fft4g.c \
    saweui/propertiesselection.cpp \
    saweui/propertiesstroke.cpp \
    sawe-brushtool.cpp \ 
    sawe-movetool.cpp \ 
    sawe-selectiontool.cpp \
    sawe-infotool.cpp
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
    signal-operationcache.h \
    signal-microphonerecorder.h \
    signal-operation.h \
    signal-operation-basic.h \
    signal-operation-composite.h \
    signal-samplesintervaldescriptor.h \
    signal-playback.h \
    signal-sink.h \
    signal-sinksource.h \
    signal-postsink.h \
    layer.h \
    tfr-stft.h \
    tfr-cwt.h \
    tfr-filter.h \
    tfr-inversecwt.h \
    tfr-chunk.h \
    tfr-chunksink.h \
    sawe-application.h \
    sawe-csv.h \
    sawe-hdf5.h \
    sawe-matlaboperation.h \
    sawe-matlabfilter.h \
    sawe-mainplayback.h \
    sawe-project.h \
    sawe-timelinewidget.h \
    signal-filteroperation.h \
    signal-worker.h \
    signal-writewav.h \
    heightmap-renderer.h \
    heightmap-collection.h \
    heightmap-reference.h \
    saweui/propertiesselection.h \
    saweui/propertiesstroke.h \
    sawe-basictool.h \
    sawe-brushtool.h \ 
    sawe-movetool.h \ 
    sawe-selectiontool.h \
    sawe-infotool.h
FORMS += mainwindow.ui \
    saweui/propertiesselection.ui \
    saweui/propertiesstroke.ui
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

# Make shaders show up in project file list in Visual Studio
win32 { 
    othersources.input = OTHER_SOURCES
    othersources.output = ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_UNIX_COMPILERS += othersources
}


####################
# Build settings

unix:IS64 = $$system(if [ -n "`uname -m | grep x86_64`" ];then echo 64; fi)
unix:DEFINES += SONICAWE_BRANCH="\'$$system(if [ -f .git/HEAD ];then cat .git/HEAD | sed -E "s/ref:\ refs\\\/heads\\\/master// | sed -E "s/ref:\ refs\\\/heads\\\///"; fi)\'"

INCLUDEPATH += ../gpumisc

unix:!macx {
INCLUDEPATH += \
    /usr/local/cuda/include
LIBS = -lsndfile \
    -lGLEW \
    -lGLU \
    -lGL \
    -lglut \
    -lportaudiocpp -lportaudio \
    -lhdf5 -lhdf5_hl \
    -L../gpumisc -lgpumisc
}

macx {
INCLUDEPATH += /usr/local/cuda/include \
    ../../libs/include \
    ../../libs/hdf5/include \
    ../../libs/zlib/include 
LIBS = -lsndfile \
    -L/usr/local/cuda/lib \
    -framework GLUT \
    -framework OpenGL \
    -L../../libs -lportaudiocpp -lportaudio \
    -L../../libs/hdf5/bin -lhdf5 -lhdf5_hl \
    -L../../libs/zlib/bin -lz \
    -L../gpumisc -lgpumisc
}

win32 {
INCLUDEPATH += \
	..\..\winlib\glut \
	..\..\winlib\glew\include \
	..\..\winlib\portaudio\include \
	..\..\winlib\libsndfile\include \
	..\..\winlib\hdf5lib\include \
	..\..\winlib\zlib\include \
	..\..\winlib
LIBS += \
	-l..\..\winlib\glut\glut32 \
	-l..\..\winlib\glew\lib\glew32 \
	-l..\..\winlib\libsndfile\libsndfile-1 \
	-l..\..\winlib\portaudio\portaudio \
	-l..\..\winlib\portaudio\portaudiocpp \
	-l..\..\winlib\hdf5lib\dll\hdf5dll \
	-l..\..\winlib\hdf5lib\dll\hdf5_hldll \
	-L..\..\winlib\boostlib
}


####################
# Temporary output

win32:RCC_DIR = tmp
MOC_DIR = tmp
OBJECTS_DIR = tmp/
UI_DIR = tmp

CONFIG(debug, debug|release):OBJECTS_DIR = tmp/debug/
else:OBJECTS_DIR = tmp/release/

# #######################################################################
# CUDA
# #######################################################################

### CUDA Compiler flags
emulation: CUDA_FLAGS += --device-emulation
emulation: LIBS += -lcufftemu -lcudartemu
!emulation: LIBS += -lcufft -lcudart
CONFIG(debug, debug|release): CUDA_FLAGS += -g
CUDA_FLAGS += --use_fast_math

win32 { 
    INCLUDEPATH += $(CUDA_INC_PATH)
    LIBS += -L$(CUDA_LIB_PATH)
	QMAKE_CXXFLAGS -= -Zc:wchar_t-
	QMAKE_CXXFLAGS += -Zc:wchar_t
    cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = $(CUDA_BIN_PATH)/nvcc.exe \
        -c \
        -Xcompiler \
        \"$$join(QMAKE_CXXFLAGS," ")\" \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        $$CUDA_FLAGS \
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
    cuda.output = $${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $${CUDA_DIR}/bin/nvcc \
        -c \
        -Xcompiler \
        $$join(QMAKE_CXXFLAGS,",") \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        $$CUDA_FLAGS \
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
    cuda.output = $${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $${CUDA_DIR}/bin/nvcc \
        -c \
        -Xcompiler \
        $$join(QMAKE_CXXFLAGS,",") \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        $$CUDA_FLAGS \
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
