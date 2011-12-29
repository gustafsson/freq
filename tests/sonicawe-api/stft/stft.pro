#-------------------------------------------------
#
# Project created by Johan 2011-12-09
#
#-------------------------------------------------

####################
# Build settings

QT += testlib
QT += opengl

CONFIG   += console
win32:CONFIG += debug_and_release
macx:CONFIG   -= app_bundle

TEMPLATE = app
win32:TEMPLATE = vcapp


macx:QMAKE_LFLAGS += -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -m32 -arch i386
macx:QMAKE_CXXFLAGS += -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -m32 -arch i386 -Wfatal-errors
macx:QMAKE_CFLAGS += -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -m32 -arch i386 -Wfatal-errors
unix:QMAKE_CXXFLAGS_RELEASE += -fopenmp
unix:QMAKE_LFLAGS_RELEASE += -fopenmp
win32:QMAKE_CXXFLAGS_RELEASE += /openmp


####################
# Source code

SOURCES += *.cpp


####################
# Compiler flags

DEFINES += SRCDIR=\\\"$$PWD/\\\"

unix:IS64 = $$system(if [ "`uname -m`" = "x86_64" ]; then echo 64; fi)

win32 {
    QMAKE_CXXFLAGS_RELEASE += /openmp
    QMAKE_CXXFLAGS += /MP
    DEFINES += _SCL_SECURE_NO_WARNINGS _CRT_SECURE_NO_WARNINGS
    QMAKE_CXXFLAGS_DEBUG -= /Zi
    QMAKE_CXXFLAGS_DEBUG += /ZI
    QMAKE_LFLAGS_DEBUG += /OPT:NOICF /OPT:NOREF
    QMAKE_LFLAGS_DEBUG += \
        /NODEFAULTLIB:LIBCPMT \ # LIBCPMT is linked by boost_serialization but we don't want it to, this row is required to link successfully
        /NODEFAULTLIB:LIBCMT \ # some other lib links LIBCMT and MSVCRT too, but LINK.EXE ignores them even without explicit NODEFAULTLIB
        /NODEFAULTLIB:MSVCRT
    QMAKE_LFLAGS_RELEASE += \
        /NODEFAULTLIB:LIBCPMT \ # LIBCPMT is linked by boost_serialization but we don't want it to, this row is required to link successfully
        /NODEFAULTLIB:LIBCMT # some other lib links LIBCMT too, but LINK.EXE ignores it even without explicit NODEFAULTLIB
}


INCLUDEPATH += \
    ../../../../../sonic/gpumisc \
    ../../../../../sonic/sonicawe \

	
win32 {
    INCLUDEPATH += \
        ../../../../../winlib/glut \
        ../../../../../winlib/glew/include \
        ../../../../../winlib \

    LIBS += \
        -l../../../../../winlib/glut/glut32 \
        -l../../../../../winlib/glew/lib/glew32 \
        -L../../../../../winlib/boostlib \

    LIBS += \
        -L../../../../sonicawe/release -lsonicawe \
        -L../../../../gpumisc/release -lgpumisc \
		
}

# build sonicawe with qmake CONFIG+=testlib
unix:LIBS += \
        -L../../../../sonicawe -lsonicawe \

# find libsonicawe when executing from project path
unix:!macx:QMAKE_LFLAGS += -Wl,-rpath=../../../

macx:INCLUDEPATH += \
        ../../../../../maclib/boost_1_45_0 \


####################
# Temporary output

win32:RCC_DIR = tmp
MOC_DIR = tmp
OBJECTS_DIR = tmp/
UI_DIR = tmp

CONFIG(debug, debug|release):OBJECTS_DIR = tmp/debug/
else:OBJECTS_DIR = tmp/release/
