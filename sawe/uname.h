#ifndef UNAME_H
#define UNAME_H

#define STRINGIFY(x) #x
#define TOSTR(x) STRINGIFY(x)

#ifdef SONICAWE_UNAME
    #ifdef SONICAWE_UNAMEm
        #define UNAME3 TOSTR(SONICAWE_UNAME) " " TOSTR(SONICAWE_UNAMEm)
    #else
        #define UNAME3 TOSTR(SONICAWE_UNAME)
    #endif
    #ifdef SONICAWE_DISTCODENAME
        #define UNAME2 UNAME3 " " TOSTR(SONICAWE_DISTCODENAME)
    #else
        #define UNAME2 UNAME3
    #endif
#else
    #ifdef _DEBUG
        #define UNAME2 "debug build, undefined platform"
    #else
        #define UNAME2 "undefined platform"
    #endif
#endif

#ifdef USE_CUDA
    #define COMPUTING_PLATFORM "CUDA"
#elif defined(USE_OPENCL)
    #define COMPUTING_PLATFORM "OPENCL"
#else
    #define COMPUTING_PLATFORM "CPU"
#endif

#define UNAME UNAME2 " " COMPUTING_PLATFORM

#endif // UNAME_H
