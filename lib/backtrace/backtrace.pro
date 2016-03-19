# This builds a static library
# Use Makefile.unittest to build and run a unit test

TARGET = backtrace
TEMPLATE = lib
win32-msvc*:TEMPLATE = vclib
win32-msvc*:CONFIG += debug_and_release

CONFIG += staticlib warn_on
CONFIG += c++11 buildflags
CONFIG += tmpdir
CONFIG += precompile_header_with_all_headers

QT += opengl

SOURCES += *.cpp
HEADERS += *.h

win32: INCLUDEPATH += ../../3rdparty/windows
win32-msvc*: SOURCES += windows/msvc/*.cpp
win32-msvc*: HEADERS += windows/msvc/*.h

macx:exists(/opt/local/include/): INCLUDEPATH += /opt/local/include/ # macports
macx:exists(/usr/local/include/): INCLUDEPATH += /usr/local/include/ # homebrew

OTHER_FILES += \
    LICENSE \
    README.md \
    *.pro \

win32 { 
    othersources.input = OTHER_FILES
    othersources.output = ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_COMPILERS += othersources
}
