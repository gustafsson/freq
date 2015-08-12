# -------------------------------------------------
# Project created by QtCreator 2009-11-08T21:47:47
# -------------------------------------------------


####################

TARGET = gpumisc
TEMPLATE = lib
win32:TEMPLATE = vclib
win32:CONFIG += debug_and_release

CONFIG += staticlib warn_on
CONFIG += c++11 buildflags
CONFIG += tmpdir
CONFIG += precompile_header_with_all_headers
CONFIG += legacy-opengl

QT += opengl # to use QGLWidget
QT += widgets # to use QApplication

DEFINES += GPUMISC_LIBRARY

SOURCES += \
    cpumemorystorage.cpp \
    cpuproperties.cpp \
    datastorage.cpp \
    datastoragestring.cpp \
    debugbuf.cpp \
    factor.cpp \
    float16.cpp \
    geometricalgebra.cpp \
    GlException.cpp \
    glframebuffer.cpp \
    glinfo.cpp \
    glprojection.cpp \
    glPushContext.cpp \
    glsyncobjectmutex.cpp \
    GlTexture.cpp \
    gltextureread.cpp \
    gluerrorstring.cpp \
    gluinvertmatrix.cpp \
    gluperspective.cpp \
    gluproject_ios.cpp \
    gluunproject.cpp \
    largememorypool.cpp \
    mappedvbovoid.cpp \
    neat_math.cpp \
    redirectstdout.cpp \
    resampletexture.cpp \
    ThreadChecker.cpp \
    unittest.cpp \
    vbo.cpp \

HEADERS += \
    computationkernel.h \
    cpumemoryaccess.h \
    cpumemorystorage.h \
    datastorage.h \
    datastorageaccess.h \
    datastoragestring.h \
    cpuproperties.h \
    debugbuf.h \
    debugmacros.h \
    debugstreams.h \
    deprecated.h \
    factor.h \
    float16.h \
    gl.h \
    GlException.h \
    glframebuffer.h \
    glinfo.h \
    glprojection.h \
    glPushContext.h \
    glsyncobjectmutex.h \
    GlTexture.h \
    gltextureread.h \
    GLvector.h \
    geometricalgebra.h \
    gluerrorstring.h \
    gluinvertmatrix.h \
    gluperspective.h \
    gluproject_ios.h \
    gluunproject.h \
    gpumisc_global.h \
    HasSingleton.h \
    InvokeOnDestruction.hpp \
    largememorypool.h \
    mappedvbo.h \
    mappedvbovoid.h \
    msc_stdc.h \
    neat_math.h \
    operate.h \
    printmatrix.h \
    redirectstdout.h \
    redirectStream.h \
    releaseaftercontext.h \
    resample.h \
    resamplecpu.h \
    resamplehelpers.h \
    resampletexture.h \
    resampletypes.h \
    Statistics.h \
    StatisticsRandom.h \
    TAni.h \
    ThreadChecker.h \
    throwInvalidArgument.h \
    tmatrix.h \
    tmatrixstring.h \
    tvector.h \
    tvectorstring.h \
    unittest.h \
    unsignedf.h \
    vbo.h \


win32: INCLUDEPATH += \
    ../sonicawe-winlib \
    ../sonicawe-winlib/glew/include

macx:exists(/opt/local/include/): INCLUDEPATH += /opt/local/include/ # macports
macx:exists(/usr/local/include/): INCLUDEPATH += /usr/local/include/ # homebrew

INCLUDEPATH += ../backtrace

OTHER_FILES += \
    LICENSE \
    *.pro \

win32 { 
    othersources.input = OTHER_FILES
    othersources.output = ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_COMPILERS += othersources
}
