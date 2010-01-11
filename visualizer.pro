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
    wavelett.cpp \
    spectrogram.cpp \
    transform-inverse.cpp \
    filter.cpp \
    transform-chunk.cpp \
    transform.cpp \
    waveform.cpp
HEADERS += mainwindow.h \
    displaywidget.h \
    transformdata.h \
    waveform.h \
    spectrogram.h \
    transform-inverse.h \
    filter.h \
    transform-chunk.h \
    transform.h \
    wavelet.cu.h
FORMS += mainwindow.ui
OTHER_FILES += wavelet.cu \
    wavelet.cu
