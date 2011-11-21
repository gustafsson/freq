#-------------------------------------------------
#
# Project created by QtCreator 2011-06-04T18:20:23
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = masher
CONFIG   += console
macx:CONFIG   -= app_bundle

TEMPLATE = app
win32:TEMPLATE = vcapp
win32:CONFIG += debug_and_release

SOURCES += main.cpp

OBJECTS_DIR = tmp/

