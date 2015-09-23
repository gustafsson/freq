# This builds a static library
# Use Makefile.unittest to build and run a unit test

TARGET = heightmap
TEMPLATE = lib
win32:TEMPLATE = vclib
win32:CONFIG += debug_and_release

CONFIG += staticlib warn_on
CONFIG += c++11 buildflags
CONFIG += tmpdir
CONFIG += precompile_header_with_all_headers

# fonts for RenderAxes
#CONFIG += freetype-gl # the embedded font works just fine
#DEFINES += USE_GLUT

QT += opengl

RESOURCES += \
    heightmap/render/shaders.qrc \

PWD = $$_PRO_FILE_PWD_
SAWEROOT = $$PWD/../..

SOURCES += \
    $$PWD/heightmap/*.cpp \
    $$PWD/heightmap/blockmanagement/*.cpp \
    $$PWD/heightmap/blockmanagement/merge/*.cpp \
    $$PWD/heightmap/render/*.cpp \

HEADERS += \
    $$PWD/heightmap/*.h \
    $$PWD/heightmap/blockmanagement/*.h \
    $$PWD/heightmap/blockmanagement/merge/*.h \
    $$PWD/heightmap/render/*.h \

INCLUDEPATH += ../backtrace ../gpumisc ../signal
win32: INCLUDEPATH += ../sonicawe-winlib

macx:exists(/opt/local/include/): INCLUDEPATH += /opt/local/include/ # macports
macx:exists(/usr/local/include/): INCLUDEPATH += /usr/local/include/ # homebrew

OTHER_FILES += \
    *.pro \

win32 { 
    othersources.input = OTHER_FILES
    othersources.output = ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_COMPILERS += othersources
}
