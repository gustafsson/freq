#ifndef UNUSED_H
#define UNUSED_H

// avoid warning: unused variable
#ifdef _WIN32
#define UNUSED(X) X
#else
#define UNUSED(X) X __attribute__ ((unused))
#endif

#endif // UNUSED_H
