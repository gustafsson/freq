# -------------------------------------------------
# Project created by QtCreator 2009-11-06T11:26:14
# -------------------------------------------------


# features directory
qtfeatures = ../qtfeatures


####################
# Project settings

TARGET = sonicawe
!win32:customtarget {
    # Changing the target would also change the name of the genereated VC++ project file which would break the configuration in the .sln-file. Therefore, in windows, rename the generated executable afterwards instead.
    TARGET = $$CUSTOMTARGET
}

customtarget {
    DEFINES += TARGETNAME=$${TARGETNAME}
}

testlib {
    TEMPLATE = lib
    win32:TEMPLATE = vclib
    CONFIG += sharedlib
    DEFINES += SAWE_EXPORTDLL
} else {
    DEFINES += SAWE_NODLL
    TEMPLATE = app
    win32:TEMPLATE = vcapp
    win32:CONFIG -= embed_manifest_dll
    win32:CONFIG += embed_manifest_exe
}


#CONFIG += $$qtfeatures/buildflags
#CONFIG += console # console output
############################
win32:CONFIG += debug_and_release
macx:CONFIG -= app_bundle
CONFIG += warn_on

macx {
    macosx105 {
        QMAKE_LFLAGS += -isysroot /Developer/SDKs/MacOSX10.5.sdk
        QMAKE_CXXFLAGS += -isysroot /Developer/SDKs/MacOSX10.5.sdk
        QMAKE_CFLAGS += -isysroot /Developer/SDKs/MacOSX10.5.sdk
    }
    macosx32bit {
        QMAKE_LFLAGS += -m32 -arch i386
        QMAKE_CXXFLAGS += -m32 -arch i386
        QMAKE_CFLAGS += -m32 -arch i386
    }
    QMAKE_CXXFLAGS += -Wfatal-errors
    QMAKE_CFLAGS += -Wfatal-errors
}

unix:QMAKE_CXXFLAGS_RELEASE += -fopenmp
unix:QMAKE_LFLAGS_RELEASE += -fopenmp
win32:QMAKE_CXXFLAGS_RELEASE += /openmp
unix:!macx:QMAKE_CXXFLAGS_DEBUG += -ggdb
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

#gold-linker {
#    QMAKE_LINK = gold
#    QMAKE_LINK = ld
# TODO add system libraries which are included by g++-4.3, but not by 'gold' (nor 'ld'), could use output from compiling with "g++-4.3 -v"
#    QMAKE_LFLAGS += -L/usr/lib -L/usr/X11R6/lib -shared-libgcc -mtune=generic /usr/lib/gcc/x86_64-linux-gnu/4.3.4/collect2 --build-id --eh-frame-hdr -m elf_x86_64 --hash-style=both -dynamic-linker /lib64/ld-linux-x86-64.so.2 -o sonicawe -z relro /usr/lib/gcc/x86_64-linux-gnu/4.3.4/../../../../lib/crt1.o /usr/lib/gcc/x86_64-linux-gnu/4.3.4/../../../../lib/crti.o /usr/lib/gcc/x86_64-linux-gnu/4.3.4/crtbegin.o -L/usr/local/cuda/lib64 -L/usr/lib -L/usr/X11R6/lib -L../lib/gpumisc -L/usr/lib/gcc/x86_64-linux-gnu/4.3.4 -L/usr/lib/gcc/x86_64-linux-gnu/4.3.4 -L/usr/lib/gcc/x86_64-linux-gnu/4.3.4/../../../../lib -L/lib/../lib -L/usr/lib/../lib -L/usr/lib/gcc/x86_64-linux-gnu/4.3.4/../../.. -L/usr/lib/x86_64-linux-gnu
#    QMAKE_LFLAGS += -rpath=/usr/share/sonicawe/
#}

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
    sawe/configuration/configuration.cpp \

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
    tools/dropnotifyform.ui \
    tools/sendfeedback.ui \
    tools/commands/commandhistory.ui \
    tools/splashscreen.ui \

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
#    sawe/configuration/configuration.cpp

# "Other files" for Qt Creator
OTHER_FILES += \
    $$CUDA_SOURCES \
    $$SHADER_SOURCES \
    $$CONFIGURATION_SOURCES \
    sonicawe.rc \

# "Other files" for Visual Studio
OTHER_FILES_VS += \
    $$SHADER_SOURCES \
    *.pro \

CONFIG += $$qtfeatures/otherfilesvs


####################
# Build settings
CONFIG += $$qtfeatures/sawelibs
QT += opengl
QT += network
DEFINES += SAWE_NO_MUTEX
#DEFINES += CUDA_MEMCHECK_TEST


####################
# Temporary output

TMPDIR=
customtarget: TMPDIR=target/$${TARGETNAME}
testlib {
    TMPDIR = lib/$${TMPDIR}
}

CONFIG += $$qtfeatures/tmpdir


# #######################################################################
# OpenCL
# #######################################################################
useopencl {
    SOURCES += \
        tfr/clfft/*.cpp

    HEADERS += \
        tfr/clfft/*.h

    CONFIG += $$qtfeatures/opencl
}


# #######################################################################
# CUDA
# #######################################################################
usecuda: CONFIG += $$qtfeatures/cuda


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
unix:testlib:CONFIGURATION_FLAGS += -fPIC
configuration.commands = $${QMAKE_CXX} \
    $${CONFIGURATION_FLAGS} \
    $$join(CONFIGURATION_DEFINES,'" -D"','-D"','"') \
    $$join(DEFINES,'" -D"','-D"','"') \
    $(INCPATH) \
    -c ${QMAKE_FILE_IN} \
    $${CXX_OUTPARAM}"${QMAKE_FILE_OUT}"
QMAKE_EXTRA_COMPILERS += configuration
