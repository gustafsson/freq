#ifndef SIGNALDLL_H
#define SIGNALDLL_H

#if defined(SIGNAL_NODLL) || !defined(_MSC_VER)
 #define SignalDll
#else
 #if defined(SIGNAL_EXPORTDLL)
  #define SignalDll __declspec( dllexport )
 #else
  #define SignalDll __declspec( dllimport )
 #endif
#endif

#endif // SIGNALDLL_H
