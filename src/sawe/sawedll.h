#ifndef SAWEDLL_H
#define SAWEDLL_H

#if defined(SAWE_NODLL) || !defined(_MSC_VER)
 #define SaweDll
#else
 #if defined(SAWE_EXPORTDLL)
  #define SaweDll __declspec( dllexport )
 #else
  #define SaweDll __declspec( dllimport )
 #endif
#endif

#endif // SAWEDLL_H
