#-------------------------------------------------
#
# Project created by Johan 2011-12-09
#
#-------------------------------------------------

####################
# Compiler settings

QT += testlib
QT += opengl

CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


unix:QMAKE_CXXFLAGS_RELEASE += -fopenmp
unix:QMAKE_LFLAGS_RELEASE += -fopenmp
win32:QMAKE_CXXFLAGS_RELEASE += /openmp


####################
# Source code


SOURCES += *.cpp

DEFINES += SRCDIR=\\\"$$PWD/\\\"

unix:IS64 = $$system(if [ "`uname -m`" = "x86_64" ]; then echo 64; fi)

INCLUDEPATH += \
    ../../../../sonic/gpumisc \
    ../../../../sonic/sonicawe \

# build sonicawe with qmake CONFIG+=testlib
LIBS = -L../../../sonicawe -lsonicawe

# find libsonicawe when executing from project path
QMAKE_LFLAGS += -Wl,-rpath=../../

####################
# Temporary output

win32:RCC_DIR = tmp
MOC_DIR = tmp
OBJECTS_DIR = tmp/
UI_DIR = tmp

CONFIG(debug, debug|release):OBJECTS_DIR = tmp/debug/
else:OBJECTS_DIR = tmp/release/

