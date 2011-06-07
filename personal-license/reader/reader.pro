#-------------------------------------------------
#
# Project created by QtCreator 2011-06-04T19:20:50
#
#-------------------------------------------------

QT       -= core gui

TARGET = reader
TEMPLATE = lib
CONFIG += staticlib warn_on

DEFINES += READER_LIBRARY

SOURCES += reader.cpp

HEADERS += reader.h\
        reader_global.h

QMAKE_CXXFLAGS += -O3

OBJECTS_DIR = tmp/

