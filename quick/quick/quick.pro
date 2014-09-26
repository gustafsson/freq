TEMPLATE = app
TARGET = Freq
QT += qml quick widgets multimedia

CONFIG += c++11

SOURCES += *.cpp
HEADERS += *.h

RESOURCES += qml.qrc
!win32:ICON = freq.icns
win32:RC_ICONS = freq.ico
# http://qt-project.org/forums/viewthread/34652
BUNDLE_DATA.files = $$system("ls $$_PRO_FILE_PWD_/iOS_BundleData/*.png")
QMAKE_BUNDLE_DATA += BUNDLE_DATA
QMAKE_INFO_PLIST = $$_PRO_FILE_PWD_/ios/Info.plist

# Additional import path used to resolve QML modules in Qt Creator's code model
QML_IMPORT_PATH =

#INCLUDEPATH += /Users/johan/dev/sonicawe/lib/sonicawe-ios
# how to find a feature in a shadow build?
#PWD = $$_PRO_FILE_PWD_
SAWEROOT = $$_PRO_FILE_PWD_/../..
CONFIG += buildflags
#QMAKEFEATURES = $$_PRO_FILE_PWD_/../../features

QT += opengl
QT += network

INCLUDEPATH += \
    $$SAWEROOT/lib/gpumisc \
    $$SAWEROOT/lib/backtrace \
    $$SAWEROOT/lib/justmisc \
    $$SAWEROOT/lib/signal \
    $$SAWEROOT/lib/tfr \
    $$SAWEROOT/lib/heightmap \
    $$SAWEROOT/lib/tfrheightmap \
    $$SAWEROOT/lib/heightmapview \
    $$SAWEROOT/src \

LIBS += \
    -L../../lib/justmisc -ljustmisc \
    -L../../lib/backtrace -lbacktrace \
    -L../../lib/gpumisc -lgpumisc \
    -L../../lib/signal -lsignal \
    -L../../lib/tfr -ltfr \
    -L../../lib/heightmap -lheightmap \
    -L../../lib/tfrheightmap -ltfrheightmap \
    -L../../lib/heightmapview -lheightmapview \

macx-ios*: LIBS += -F$$SAWEROOT/3rdparty/ios/framework -framework flac
!macx-ios*: LIBS += -framework GLUT -L/usr/local/lib -lFLAC

macx:exists(/opt/local/include/): INCLUDEPATH += /opt/local/include/ # macports
macx:exists(/usr/local/include/): INCLUDEPATH += /usr/local/include/ # homebrew

# Default rules for deployment.
include(deployment.pri)
