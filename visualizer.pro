# -------------------------------------------------
# Project created by QtCreator 2009-11-06T11:26:14
# -------------------------------------------------
QT += opengl \
    testlib
TARGET = visualizer
TEMPLATE = app
SOURCES += main.cpp \
    mainwindow.cpp \
    displaywidget.cpp \
    wavelettransform.cpp \
    transformdata.cpp \
    waveform.cpp
HEADERS += mainwindow.h \
    displaywidget.h \
    wavelettransform.h \
    transformdata.h \
    waveform.h
FORMS += mainwindow.ui
OTHER_FILES += wavelett.cu
