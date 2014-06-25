# This builds a static library
# Use Makefile.unittest to build and run a unit test

TARGET = signal
TEMPLATE = lib
win32:TEMPLATE = vclib
win32:CONFIG += debug_and_release

CONFIG += staticlib warn_on
CONFIG += c++11 buildflags

QT += opengl

SOURCES += \
    signal/*.cpp \
    signal/processing/*.cpp \
    test/*.cpp \

HEADERS += \
    signal/*.h \
    signal/processing/*.h \
    test/*.h \

INCLUDEPATH += ../backtrace ../gpumisc
win32: INCLUDEPATH += ../sonicawe-winlib

macx:exists(/opt/local/include/): INCLUDEPATH += /opt/local/include/ # macports
macx:exists(/usr/local/include/): INCLUDEPATH += /usr/local/include/ # homebrew

MOC_DIR = tmp
OBJECTS_DIR = tmp/
CONFIG(debug, debug|release):OBJECTS_DIR = $${OBJECTS_DIR}debug/
else:OBJECTS_DIR = $${OBJECTS_DIR}release/


OTHER_FILES += \
    LICENSE \
    *.pro \

win32 { 
    othersources.input = OTHER_FILES
    othersources.output = ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_COMPILERS += othersources
}
