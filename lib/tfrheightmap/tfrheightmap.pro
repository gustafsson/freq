# This builds a static library
# Use Makefile.unittest to build and run a unit test

TARGET = tfrheightmap
TEMPLATE = lib
win32:TEMPLATE = vclib
win32:CONFIG += debug_and_release

CONFIG += staticlib warn_on
CONFIG += c++11 buildflags
CONFIG += tmpdir

QT += opengl

RESOURCES += \
    heightmap/update/opengl/updateshaders.qrc \

PWD = $$_PRO_FILE_PWD_

SOURCES += \
    $$PWD/heightmap/*.cpp \
    $$PWD/heightmap/tfrmappings/*.cpp \
    $$PWD/heightmap/update/*.cpp \
    $$PWD/heightmap/update/opengl/*.cpp \

HEADERS += \
    $$PWD/heightmap/*.h \
    $$PWD/heightmap/tfrmappings/*.h \
    $$PWD/heightmap/update/*.h \
    $$PWD/heightmap/update/opengl/*.h \

INCLUDEPATH += ../backtrace ../gpumisc ../signal ../tfr ../justmisc ../heightmap
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
