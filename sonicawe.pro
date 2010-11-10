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
CONFIG += console # console output
QT += opengl

unix:QMAKE_CXXFLAGS_DEBUG += -ggdb
!win32:QMAKE_CXXFLAGS_RELEASE -= -O2
!win32:QMAKE_CXXFLAGS_RELEASE += -O3
win32:QMAKE_LFLAGS += /FORCE:MULTIPLE
QMAKE_CXXFLAGS_DEBUG += -D_DEBUG

!macx&!win32: QMAKE_CXX = colorgcc
#macx:QMAKE_CXX = g++ # Should not need this macx: with !macx&!win32 above

### Settings for using llvm instead of gcc on linux
llvm {
    QMAKE_CXX = llvm-g++
    QMAKE_CC = llvm-gcc
    QMAKE_LINK = llvm-g++
}


####################
# Source code

RESOURCES += \
    ui/icon-resources.qrc \

SOURCES += \
    adapters/audiofile.cpp \
    adapters/csv.cpp \
    adapters/hdf5.cpp \
    adapters/matlabfilter.cpp \
    adapters/matlaboperation.cpp \
    adapters/microphonerecorder.cpp \
    adapters/playback.cpp \
    adapters/writewav.cpp \
    filters/*.cpp \
    heightmap/*.cpp \
    sawe/application.cpp \
    sawe/layer.cpp \
    sawe/main.cpp \
    sawe/project.cpp \
    signal/buffersource.cpp \
    signal/operation.cpp \
    signal/operation-basic.cpp \
    signal/operationcache.cpp \
    signal/postsink.cpp \
    signal/intervals.cpp \
    signal/sinksource.cpp \
    signal/source.cpp \
    signal/worker.cpp \
    tfr/chunk.cpp \
    tfr/complexbuffer.cpp \
    tfr/cwt.cpp \
    tfr/cwtchunk.cpp \
    tfr/cwtfilter.cpp \
    tfr/fft4g.c \
    tfr/filter.cpp \
    tfr/stft.cpp \
    tfr/stftfilter.cpp \
    tools/*.cpp \
    tools/support/*.cpp \
    tools/selections/*.cpp \
    ui/comboboxaction.cpp \
    ui/mainwindow.cpp \
    ui/mousecontrol.cpp \
    ui/propertiesselection.cpp \
    ui/propertiesstroke.cpp \
    ui/updatewidgetsink.cpp \

HEADERS += \
    adapters/audiofile.h \
    adapters/csv.h \
    adapters/hdf5.h \
    adapters/matlabfilter.h \
    adapters/matlaboperation.h \
    adapters/microphonerecorder.h \
    adapters/playback.h \
    adapters/writewav.h \
    filters/*.h \
    heightmap/*.h \
    sawe/application.h \
    sawe/layer.h \
    sawe/mainplayback.h \
    sawe/project.h \
    signal/buffersource.h \
    signal/operation.h \
    signal/operation-basic.h \
    signal/operationcache.h \
    signal/postsink.h \
    signal/intervals.h \
    signal/sink.h \
    signal/sinksource.h \
    signal/source.h \
    signal/worker.h \
    tfr/chunk.h \
    tfr/complexbuffer.h \
    tfr/cwt.h \
    tfr/cwtchunk.h \
    tfr/cwtfilter.h \
    tfr/filter.h \
    tfr/freqaxis.h \
    tfr/stft.h \
    tfr/transform.h \
    tfr/wavelet.cu.h \
    tfr/stftfilter.h \
    tools/*.h \
    tools/support/*.h \
    tools/selections/*.h \
    ui/comboboxaction.h \
    ui/mainwindow.h \
    ui/mousecontrol.h \
    ui/propertiesselection.h \
    ui/propertiesstroke.h \
    ui/updatewidgetsink.h \

FORMS += \
    tools/selectionviewmodel.ui \
    ui/mainwindow.ui \
    ui/propertiesselection.ui \
    ui/propertiesstroke.ui \

CUDA_SOURCES += \
    filters/*.cu \
    heightmap/block.cu \
    heightmap/resampletest.cu \
    heightmap/resampletest2.cu \
    heightmap/slope.cu \
    tfr/wavelet.cu \
    tools/support/brushfilter.cu \
    tools/support/brushpaint.cu \

SHADER_SOURCS += \
    heightmap/heightmap.frag \
    heightmap/heightmap.vert \

# "Other files" for Qt Creator
OTHER_FILES += \
    $$CUDA_SOURCES \
    $$SHADER_SOURCS \ 

# "Other files" for Visual Studio
OTHER_SOURCES += \
    $$SHADERS \
    sonicawe.pro \

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

INCLUDEPATH += \
    ../../sonic/gpumisc \
    ../../sonic/sonicawe \

unix:!macx {
LIBS = \
    -lsndfile \
    -lGLEW \
    -lGLU \
    -lGL \
    -lboost_serialization \
    -lglut \
    -lportaudiocpp -lportaudio \
    -lhdf5 -lhdf5_hl \
    -L../gpumisc -lgpumisc
}

macx {
INCLUDEPATH += \
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

LIBS += -lcufft -lcudart -lcuda
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
unix:!macx {
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
        $$join(INCLUDEPATH,'" -I "../../sonic/sonicawe/','-I "../../sonic/sonicawe/','"') \
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
