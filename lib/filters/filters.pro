# This builds a static library
# Use Makefile.unittest to build and run a unit test

TARGET = filters
TEMPLATE = lib
win32-msvc*:TEMPLATE = vclib
win32-msvc*:CONFIG += debug_and_release

CONFIG += staticlib warn_on
CONFIG += c++11 buildflags
CONFIG += tmpdir

QT += widgets

PWD = $$_PRO_FILE_PWD_

SOURCES += \
    $$PWD/filters/*.cpp \
    $$PWD/filters/support/*.cpp \

HEADERS += \
    $$PWD/filters/*.h \
    $$PWD/filters/support/*.h \

INCLUDEPATH += ../backtrace ../gpumisc ../signal ../tfr ../justmisc
win32: INCLUDEPATH += ../../3rdparty/windows

macx:exists(/opt/local/include/): INCLUDEPATH += /opt/local/include/ # macports
macx:exists(/usr/local/include/): INCLUDEPATH += /usr/local/include/ # homebrew

OTHER_FILES += \
    *.pro \

win32 { 
    othersources.input = OTHER_FILES
    othersources.output = ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_COMPILERS += othersources
}
