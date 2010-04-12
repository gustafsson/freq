# -------------------------------------------------
# Project created by QtCreator 2009-11-06T11:26:14
# -------------------------------------------------
TEMPLATE = app
win32:TEMPLATE = vcapp
macx:CONFIG -= app_bundle
QT += opengl
RESOURCES += icon-resources.qrc
QMAKE_CXXFLAGS_RELEASE = -O3
TARGET = sonicawe
SOURCES += main.cpp \
    mainwindow.cpp \
    displaywidget.cpp \
    spectrogram.cpp \
    transform-inverse.cpp \
    filter.cpp \
    transform-chunk.cpp \
    transform.cpp \
    spectrogram-vbo.cpp \
    spectrogram-renderer.cpp \
    selection.cpp \
    signal-source.cpp \
    signal-audiofile.cpp \
    signal-microphonerecorder.cpp \
    signal-operation.cpp \
    signal-invalidsamplesdescriptor.cpp \
    signal-sink.cpp \
    signal-playback.cpp \
    layer.cpp
HEADERS += mainwindow.h \
    displaywidget.h \
    spectrogram.h \
    transform-inverse.h \
    filter.h \
    transform-chunk.h \
    transform.h \
    wavelet.cu.h \
    spectrogram-vbo.h \
    spectrogram-renderer.h \
    spectrogram-slope.cu.h \
    spectrogram-block.cu.h \
    filter.cu.h \
    selection.h \
    position.h \
    signal-source.h \
    signal-audiofile.h \
    signal-microphonerecorder.h \
    signal-operation.h \
    signal-invalidsamplesdescriptor.h \
    signal-playback.h \
    signal-sink.h \
    layer.h
FORMS += mainwindow.ui
OTHER_FILES += wavelet.cu \
    spectrogram.frag \
    spectrogram.vert \
    spectrogram-slope.cu \
    spectrogram-block.cu \
    filter.cu
CUDA_SOURCES += wavelet.cu \
    spectrogram-slope.cu \
    spectrogram-block.cu \
    filter.cu
OTHER_SOURCES += spectrogram.frag \
    spectrogram.vert \
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
    -L/usr/local/cuda/lib$$IS64 \
    -lGLEW \
    -lGLU \
    -lGL \
    -lglut \
    -lportaudiocpp -lportaudio
macx:INCLUDEPATH += /usr/local/cuda/include \
      ../../libs/include 
macx:LIBS = -lsndfile \
    -laudiere \
    -L/usr/local/cuda/lib \
    -lcufft \
    -L../misc \
    -lmisc \
    -framework GLUT \
    -framework OpenGL \
    -L../../libs -lportaudiocpp -lportaudio
#    tmp/wavelet_cuda.o \
#    tmp/spectrogram-slope_cuda.o \
#    tmp/spectrogram-block_cuda.o \
#    tmp/filter_cuda.o
win32:INCLUDEPATH += ..\..\glut \
	..\..\glew\include \
	..\..\portaudio\include \
	$(BOOST_PATH)
win32:LIBS += \
	-l..\..\glut\glut32 \
	-l..\..\glew\lib\glew32 \
    -l..\..\libsndfile\libsndfile-1 \
    -L$(CUDA_LIB_PATH)\..\lib \
	-l..\..\portaudio\portaudio \
	-l..\..\portaudio\portaudiocpp \
	-L$(BOOST_PATH)\lib
LIBS += -lcufft \
    -L../gpumisc \
    -lgpumisc \

MOC_DIR = tmp
OBJECTS_DIR = tmp/
UI_DIR = tmp

# #######################################################################
# CUDA
# #######################################################################
win32 { 
    INCLUDEPATH += $(CUDA_INC_PATH)\
	.
    QMAKE_LIBDIR += $(CUDA_LIB_PATH)
    LIBS += -lcudart
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
