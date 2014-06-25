#ifndef TFRDLL_H
#define TFRDLL_H

#if defined(TFR_NODLL) || !defined(_MSC_VER)
 #define TfrDll
#else
 #if defined(TFR_EXPORTDLL)
  #define TfrDll __declspec( dllexport )
 #else
  #define TfrDll __declspec( dllimport )
 #endif
#endif

#endif // TFRDLL_H
