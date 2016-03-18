#ifndef SIGNALDLL_H
#define SIGNALDLL_H

#define SIGNAL_NODLL // breaks Sonic AWE tests

#if defined(SIGNAL_NODLL) || !defined(_WIN32)
 #define SignalDll
#else
 #if defined(SIGNAL_EXPORTDLL)
  #define SignalDll __declspec( dllexport )
 #else
  #define SignalDll __declspec( dllimport )
 #endif
#endif

#endif // SIGNALDLL_H
