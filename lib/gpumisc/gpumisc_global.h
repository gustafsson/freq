#ifndef MISC_GLOBAL_H
#define MISC_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(GPUMISC_LIBRARY)
#  define MISCSHARED_EXPORT Q_DECL_EXPORT
#else
#  define MISCSHARED_EXPORT Q_DECL_IMPORT
#endif

#endif // MISC_GLOBAL_H
