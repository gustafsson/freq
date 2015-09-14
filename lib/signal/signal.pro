# This builds a static library
# Use Makefile.unittest to build and run a unit test

TARGET = signal
TEMPLATE = lib
win32:TEMPLATE = vclib
win32:CONFIG += debug_and_release

CONFIG += staticlib warn_on
CONFIG += c++11 buildflags
CONFIG += tmpdir
CONFIG += precompile_header_with_all_headers

QT += opengl

PWD = $$_PRO_FILE_PWD_

SOURCES += \
    $$PWD/signal/*.cpp \
    $$PWD/signal/processing/*.cpp \
    $$PWD/signal/qteventworker/*.cpp \
    $$PWD/signal/cvworker/*.cpp \
    $$PWD/test/*.cpp \

HEADERS += \
    $$PWD/signal/*.h \
    $$PWD/signal/processing/*.h \
    $$PWD/signal/qteventworker/*.h \
    $$PWD/signal/cvworker/*.h \
    $$PWD/test/*.h \

INCLUDEPATH += ../backtrace ../gpumisc ../justmisc
win32: INCLUDEPATH += ../sonicawe-winlib

macx:exists(/opt/local/include/): INCLUDEPATH += /opt/local/include/ # macports
macx:exists(/usr/local/include/): INCLUDEPATH += /usr/local/include/ # homebrew

OTHER_FILES += \
    LICENSE \
    *.pro \

win32 { 
    othersources.input = OTHER_FILES
    othersources.output = ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_COMPILERS += othersources
}
