TEMPLATE = app
TARGET = Freq
QT += qml quick widgets multimedia

CONFIG += c++11

PWD = $$_PRO_FILE_PWD_

SOURCES += $$PWD/src/*.cpp
HEADERS += $$PWD/src/*.h

# http://stackoverflow.com/questions/22783381/qaudiodecoder-no-service-found
# decoding with QAudioDecoder is only supported on windows (mediafoundation backend) and linux (gstreamer backend)
!macx-ios*: QTPLUGIN += qtaudio_coreaudio qtmedia_audioengine # redundant from QT += multimedia

RESOURCES += qml.qrc
macx:!macx-ios*:ICON = freq.icns
win32:RC_ICONS = freq.ico
# app icons, http://qt-project.org/forums/viewthread/34652
macx-ios*:BUNDLE_DATA.files = $$system("ls $$_PRO_FILE_PWD_/iOS_BundleData/*.png")
macx-ios*:QMAKE_BUNDLE_DATA += BUNDLE_DATA
macx-ios*:QMAKE_INFO_PLIST = $$_PRO_FILE_PWD_/ios/Info.plist

# Additional import path used to resolve QML modules in Qt Creator's code model
QML_IMPORT_PATH =

# how to find a feature in a shadow build?
#PWD = $$_PRO_FILE_PWD_
SAWEROOT = $$_PRO_FILE_PWD_/../..
CONFIG += tmpdir buildflags
CONFIG += precompile_header_with_all_headers
#QMAKEFEATURES = $$_PRO_FILE_PWD_/../../features

QT += opengl
QT += network

INCLUDEPATH += \
    $$SAWEROOT/lib/gpumisc \
    $$SAWEROOT/lib/backtrace \
    $$SAWEROOT/lib/justmisc \
    $$SAWEROOT/lib/signal \
    $$SAWEROOT/lib/tfr \
    $$SAWEROOT/lib/filters \
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
    -L../../lib/filters -lfilters \
    -L../../lib/heightmap -lheightmap \
    -L../../lib/tfrheightmap -ltfrheightmap \
    -L../../lib/heightmapview -lheightmapview \

macx-ios*: LIBS += -F$$SAWEROOT/3rdparty/ios/framework -framework flac
!macx-ios*: LIBS += -framework GLUT -L/usr/local/lib -lFLAC

macx:exists(/opt/local/include/): INCLUDEPATH += /opt/local/include/ # macports
macx:exists(/usr/local/include/): INCLUDEPATH += /usr/local/include/ # homebrew

# Default rules for deployment.
include(deployment.pri)
