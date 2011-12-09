# -------------------------------------------------
# Project created by QtCreator 2009-11-06T11:26:14
# -------------------------------------------------


####################
# Compiler settings

TARGET = sonicawe
!win32:customtarget {
    TARGET = $$CUSTOMTARGET
}


TEMPLATE = app
win32:TEMPLATE = vcapp
testlib {
    TEMPLATE = lib
    win32:TEMPLATE = vclib
}
win32:CONFIG += debug_and_release
win32:CONFIG -= embed_manifest_dll
win32:CONFIG += embed_manifest_exe
macx:CONFIG -= app_bundle

CONFIG += warn_on
#CONFIG += console # console output
DEFINES += SAWE_NO_MUTEX
#DEFINES += CUDA_MEMCHECK_TEST
QT += opengl
QT += network

macx:QMAKE_LFLAGS += -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -m32 -arch i386
macx:QMAKE_CXXFLAGS += -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -m32 -arch i386 -Wfatal-errors
macx:QMAKE_CFLAGS += -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -m32 -arch i386 -Wfatal-errors
unix:QMAKE_CXXFLAGS_RELEASE += -fopenmp
unix:QMAKE_LFLAGS_RELEASE += -fopenmp
win32:QMAKE_CXXFLAGS_RELEASE += /openmp
unix:QMAKE_CXXFLAGS_DEBUG += -ggdb
win32:QMAKE_CXXFLAGS += /MP
!win32:QMAKE_CXXFLAGS_RELEASE -= -O2
!win32:QMAKE_CXXFLAGS_RELEASE += -O3
!win32:QMAKE_CFLAGS_RELEASE -= -O2
!win32:QMAKE_CFLAGS_RELEASE += -O3
win32:DEFINES += _SCL_SECURE_NO_WARNINGS _CRT_SECURE_NO_WARNINGS
win32:QMAKE_CXXFLAGS_DEBUG -= /Zi
win32:QMAKE_CXXFLAGS_DEBUG += /ZI
win32:QMAKE_LFLAGS_DEBUG += /OPT:NOICF /OPT:NOREF
win32:QMAKE_LFLAGS_DEBUG += \
    /NODEFAULTLIB:LIBCPMT \ # LIBCPMT is linked by boost_serialization but we don't want it to, this row is required to link successfully
    /NODEFAULTLIB:LIBCMT \ # some other lib links LIBCMT and MSVCRT too, but LINK.EXE ignores them even without explicit NODEFAULTLIB
    /NODEFAULTLIB:MSVCRT \

win32:QMAKE_LFLAGS_RELEASE += \
    /NODEFAULTLIB:LIBCPMT \ # LIBCPMT is linked by boost_serialization but we don't want it to, this row is required to link successfully
    /NODEFAULTLIB:LIBCMT \ # some other lib links LIBCMT too, but LINK.EXE ignores it even without explicit NODEFAULTLIB

QMAKE_CXXFLAGS_DEBUG += -D_DEBUG

unix:!macx: QMAKE_CXX = colorgcc

profiling {
    # Profiling with gcc, gprof doesn't work with Os X 10.5 Leopard.
    !win32:QMAKE_CXXFLAGS_RELEASE += -pg
    !win32:QMAKE_LFLAGS_RELEASE += -pg
}

### Settings for using llvm instead of gcc on linux
llvm {
    QMAKE_CXX = llvm-g++
    QMAKE_CC = llvm-gcc
    QMAKE_LINK = llvm-g++
}

gcc-4.3 {
    QMAKE_CXX = g++-4.3
    QMAKE_CC = gcc-4.3
    QMAKE_LINK = g++-4.3
}


####################
# Source code

RESOURCES += \
    ui/icon-resources.qrc \

SOURCES += \
    adapters/*.cpp \
    filters/*.cpp \
    heightmap/*.cpp \
    sawe/*.cpp \
    signal/*.cpp \
    tfr/fft4g.c \
    tfr/*.cpp \
    tools/*.cpp \
    tools/commands/*.cpp \
    tools/support/*.cpp \
    tools/selections/*.cpp \
    tools/selections/support/*.cpp \
    ui/*.cpp \

#Windows Icon
win32:SOURCES += sonicawe.rc \

HEADERS += \
    adapters/*.h \
    filters/*.h \
    heightmap/*.h \
    sawe/*.h \
    signal/*.h \
    tfr/*.h \
    tools/*.h \
    tools/commands/*.h \
    tools/support/*.h \
    tools/selections/*.h \
    tools/selections/support/*.h \
    ui/*.h \

PRECOMPILED_HEADER += sawe/project_header.h

# Qt Creator crashes every now and then in Windows if form filenames are expressed with wildcards
FORMS += \
    ui/mainwindow.ui \
    ui/propertiesselection.ui \
    ui/propertiesstroke.ui \
    tools/aboutdialog.ui \
    tools/commentview.ui \
    tools/selectionviewinfo.ui \
    tools/transforminfoform.ui \
    tools/exportaudiodialog.ui \
    tools/harmonicsinfoform.ui \
    tools/matlaboperationwidget.ui \
    tools/selections/rectangleform.ui \
    sawe/enterlicense.ui \
    tools/settingsdialog.ui \
    tools/getcudaform.ui \
    tools/sendfeedback.ui \
    tools/commands/commandhistory.ui \

CUDA_SOURCES += \
    filters/*.cu \
    heightmap/*.cu \
    tfr/*.cu \
    tools/support/*.cu \
    tools/selections/support/*.cu \

SHADER_SOURCES += \
    heightmap/heightmap.frag \
    heightmap/heightmap.vert \
    heightmap/heightmap_noshadow.vert \

CONFIGURATION_SOURCES = \
    sawe/configuration/configuration.cpp

# "Other files" for Qt Creator
OTHER_FILES += \
    $$CUDA_SOURCES \
    $$SHADER_SOURCES \
    $$CONFIGURATION_SOURCES \
    sonicawe.rc \

# "Other files" for Visual Studio
OTHER_SOURCES += \
    $$SHADER_SOURCES \
    *.pro \

# Make OTHER_SOURCES show up in project file list in Visual Studio
win32 {
    othersources.input = OTHER_SOURCES
    othersources.output = ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_COMPILERS += othersources
}

####################
# Build settings

unix:IS64 = $$system(if [ "`uname -m`" = "x86_64" ]; then echo 64; fi)

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

QMAKE_LFLAGS += -Wl,-rpath=/usr/share/sonicawe/
}

macx {
INCLUDEPATH += \
    ../../maclib/include \
    ../../maclib/boost_1_45_0 \
    ../../maclib/hdf5/include \
    ../../maclib/zlib/include \
    ../../maclib/include/sndfile
LIBS = -lsndfile \
    -L/usr/local/cuda/lib \
    -framework GLUT \
    -framework OpenGL \
    -L../../maclib -lportaudiocpp -lportaudio \
    -L../../maclib/hdf5/bin -lhdf5 -lhdf5_hl \
    -L../../maclib/zlib/lib -lz \
    -L../gpumisc -lgpumisc \
    -L../../maclib/boost_1_45_0/stage/lib \
    -lboost_serialization
}

win32 {
INCLUDEPATH += \
	../../winlib/glut \
	../../winlib/glew/include \
	../../winlib/portaudio/include \
	../../winlib/libsndfile/include \
	../../winlib/hdf5lib/include \
	../../winlib/zlib/include \
	../../winlib
LIBS += \
	-l../../winlib/glut/glut32 \
	-l../../winlib/glew/lib/glew32 \
	-l../../winlib/libsndfile/libsndfile-1 \
	-l../../winlib/hdf5lib/dll/hdf5dll \
	-l../../winlib/hdf5lib/dll/hdf5_hldll \
	-L../../winlib/boostlib
win32:QMAKE_LFLAGS_RELEASE += \
	../../winlib/portaudio/portaudio.lib \
#	../../winlib/portaudio/portaudio_x86_mt.lib \
	../../winlib/portaudio/portaudiocpp_mt.lib
win32:QMAKE_LFLAGS_DEBUG += \
	../../winlib/portaudio/portaudio.lib \
#	../../winlib/portaudio/portaudio_x86_mt_gd.lib \
	../../winlib/portaudio/portaudiocpp_mt_gd.lib
}


####################
# Temporary output

TMPDIR=
customtarget: TMPDIR=target/$${TARGETNAME}
testlib {
    TMPDIR = lib/$${TMPDIR}
}

usecuda {
  TMPDIR = $${TMPDIR}/cuda
} else:useopencl {
  TMPDIR = $${TMPDIR}/opencl
} else {
  TMPDIR = $${TMPDIR}/cpu
}

TMPDIR = tmp/$${TMPDIR}

win32:RCC_DIR = $${TMPDIR}
MOC_DIR = $${TMPDIR}
OBJECTS_DIR = $${TMPDIR}/
UI_DIR = $${TMPDIR}


CONFIG(debug, debug|release):OBJECTS_DIR = $${OBJECTS_DIR}debug/
else:OBJECTS_DIR = $${OBJECTS_DIR}release/


# #######################################################################
# Deploy configuration
# #######################################################################
CONFIGURATION_DEFINES += SONICAWE_BRANCH="$$system(git rev-parse --abbrev-ref HEAD)"
CONFIGURATION_DEFINES += SONICAWE_REVISION="$$system(git rev-parse --short HEAD)"

configuration.name = configuration
configuration.input = CONFIGURATION_SOURCES
configuration.dependency_type = TYPE_C
configuration.variable_out = OBJECTS
configuration.output = ${QMAKE_VAR_OBJECTS_DIR}${QMAKE_FILE_IN_BASE}$${first(QMAKE_EXT_OBJ)}
CONFIGURATION_FLAGS = $$QMAKE_CXXFLAGS
CONFIG(debug, debug|release):CONFIGURATION_FLAGS += $$QMAKE_CXXFLAGS_DEBUG
else:CONFIGURATION_FLAGS += $$QMAKE_CXXFLAGS_RELEASE
win32:CONFIGURATION_FLAGS += /EHsc
win32:CXX_OUTPARAM = /Fo
else:CXX_OUTPARAM = "-o "
testlib:CONFIGURATION_FLAGS+=-fPIC
configuration.commands = $${QMAKE_CXX} \
    $${CONFIGURATION_FLAGS} \
    $$join(CONFIGURATION_DEFINES,'" -D"','-D"','"') \
    $$join(DEFINES,'" -D"','-D"','"') \
    $(INCPATH) \
    -c ${QMAKE_FILE_IN} \
    $${CXX_OUTPARAM}"${QMAKE_FILE_OUT}"
QMAKE_EXTRA_COMPILERS += configuration


# #######################################################################
# OpenCL
# #######################################################################
useopencl {
DEFINES += USE_OPENCL

SOURCES += \
    tfr/clfft/*.cpp

HEADERS += \
    tfr/clfft/*.h

macx: LIBS += -framework OpenCL
!macx: LIBS += -lOpenCL

win32 {
    # use OpenCL headers from Cuda Gpu Computing SDK
    INCLUDEPATH += "$(CUDA_INC_PATH)"
    LIBS += -L"$(CUDA_LIB_PATH)"
}

unix:!macx {
    OPENCL_DIR = /usr/local/cuda
    INCLUDEPATH += $$OPENCL_DIR/include
}

macx {
    OPENCL_DIR = /usr/local/cuda
    INCLUDEPATH += $$OPENCL_DIR/include
}
}


# #######################################################################
# CUDA
# #######################################################################
usecuda {
DEFINES += USE_CUDA

LIBS += -lcufft -lcudart -lcuda
CONFIG(debug, debug|release): CUDA_FLAGS += -g
CUDA_FLAGS += --use_fast_math
#CUDA_FLAGS += --ptxas-options=-v


CUDA_CXXFLAGS = $$QMAKE_CXXFLAGS
CONFIG(debug, debug|release):CUDA_CXXFLAGS += $$QMAKE_CXXFLAGS_DEBUG
else:CUDA_CXXFLAGS += $$QMAKE_CXXFLAGS_RELEASE
win32 { 
    INCLUDEPATH += "$(CUDA_INC_PATH)"
    LIBS += -L"$(CUDA_LIB_PATH)"
    CUDA_CXXFLAGS -= -Zc:wchar_t-
    CUDA_CXXFLAGS += -Zc:wchar_t
    CUDA_CXXFLAGS += /EHsc
    cuda.output = $${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = \"$(CUDA_BIN_PATH)/nvcc.exe\" \
		-ccbin $${QMAKE_CC} \
        -c \
        -Xcompiler \
        \"$$join(CUDA_CXXFLAGS," ")\" \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        $$CUDA_FLAGS \
        "${QMAKE_FILE_NAME}" \
		-o \
        "${QMAKE_FILE_OUT}"
}
unix:!macx {
    # auto-detect CUDA path
    # CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
    CUDA_DIR = /usr/local/cuda
    INCLUDEPATH += $$CUDA_DIR/include
    QMAKE_LIBDIR += $$CUDA_DIR/lib$$IS64
    cuda.output = $${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $${CUDA_DIR}/bin/nvcc \
		-ccbin $${QMAKE_CC} \
        -c \
        -Xcompiler \
        $$join(CUDA_CXXFLAGS,",") \
        $$join(INCLUDEPATH,'" -I "../../sonic/sonicawe/','-I "../../sonic/sonicawe/','"') \
        $$CUDA_FLAGS \
        ${QMAKE_FILE_NAME} \
        -o \
        ${QMAKE_FILE_OUT}
    cuda.dependcy_type = TYPE_C
    cuda.depend_command_dosntwork = nvcc \
        -M \
        -Xcompiler \
        $$join(CUDA_CXXFLAGS,",") \
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
		-ccbin $${QMAKE_CC} \
        -c \
        -Xcompiler \
        $$join(CUDA_CXXFLAGS,",") \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        $$CUDA_FLAGS \
        ${QMAKE_FILE_NAME} \
        -o \
        ${QMAKE_FILE_OUT}
    cuda.dependcy_type = TYPE_C
}

cuda.input = CUDA_SOURCES
QMAKE_EXTRA_COMPILERS += cuda

} #usecuda
