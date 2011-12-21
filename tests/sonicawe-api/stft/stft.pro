#-------------------------------------------------
#
# Project created by Johan 2011-12-09
#
#-------------------------------------------------

####################
# Compiler settings

QT += testlib
QT += opengl

TARGET = stft
CONFIG   += console
CONFIG   -= app_bundle
win32:CONFIG += debug_and_release

TEMPLATE = app
win32:TEMPLATE = vcapp


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

INCLUDEPATH += \
    ../../../../../sonic/gpumisc \
    ../../../../../sonic/sonicawe \

win32 {
    INCLUDEPATH += \
        ../../../../../winlib/glut \
        ../../../../../winlib/glew/include \
        ../../../../../winlib \
    LIBS += \
        -l../../../../winlib/glut/glut32 \
        -l../../../../winlib/glew/lib/glew32 \

    LIBS += \
        -L../../../../sonicawe/debug -lsonicawe \
} else {
    # build sonicawe with qmake CONFIG+=testlib
    LIBS += -L../../../../sonicawe -lsonicawe \

    # find libsonicawe when executing from project path
    QMAKE_LFLAGS += -Wl,-rpath=../../../
}


####################
# Temporary output

win32:RCC_DIR = tmp
MOC_DIR = tmp
OBJECTS_DIR = tmp/
UI_DIR = tmp

CONFIG(debug, debug|release):OBJECTS_DIR = tmp/debug/
else:OBJECTS_DIR = tmp/release/

