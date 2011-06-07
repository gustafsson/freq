#ifndef READER_GLOBAL_H
#define READER_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(READER_LIBRARY)
#  define READERSHARED_EXPORT Q_DECL_EXPORT
#else
#  define READERSHARED_EXPORT Q_DECL_IMPORT
#endif

#endif // READER_GLOBAL_H
