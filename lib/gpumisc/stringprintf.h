/**
 like sprintf but outputs to a std::string instead of a char*
 printfstring is like printf but returns the result in a std::string instead of printing it to screen

 johan.b.gustafsson@gmail.com
 */
#pragma once

#ifdef _MSC_VER
#if _MSC_VER <= 1499
#ifndef _Check_return_
#define _Check_return_ __checkReturn_opt
#endif // _Check_return_
#endif // _MSC_VER <= 1499
#else
#ifndef _Check_return_
#define _Check_return_
#endif
#ifndef __in_z
#define __in_z
#endif
#define __format_string
#endif

#include <stdio.h>
#include <stdarg.h>
#include <string>
// Marcus: compiler complains about this. __cdecl och _Check_return_ makron funkar inte? VC++ only?
//template<typename t> _Check_return_ int __cdecl  
template<typename t> int 
stringprintf(std::basic_string<t>& s,  __in_z __format_string const t* f, ...);

template<typename t> std::basic_string<t> 
printfstring( __in_z __format_string const t* f, va_list va, int* result);

template<typename t> std::basic_string<t> 
printfstring( __in_z __format_string const t* f, ...);


#if defined(__cplusplus) && !defined(__CUDACC__)
// well, I do like printf... and I know it's not typesafe, but damn it's handy.
// just don't abuse the "..." syntax will you?
template<typename t> int 
ostreamprintf(std::basic_ostream<t>& o, __in_z __format_string const t* f, ...);
template<typename t> int
ostreamprintf(std::basic_ostream<t>& o, __in_z __format_string const t* f, va_list args);

#endif // if defined(__cplusplus) && !defined(__CUDACC__)
//same problem here...
// please use these for printf output rather than fprintf(stderr,...) since they can be redirected (using std::cerr.rdbuf( newbuffer );)
//extern "C"
_Check_return_ int 
coutprintf(__in_z __format_string const char* f, ...);
//extern "C"
_Check_return_ int 
clogprintf(__in_z __format_string const char* f, ...);
//extern "C"
_Check_return_ int 
cerrprintf(__in_z __format_string const char* f, ...);
//extern "C"
_Check_return_ int 
wcoutprintf(__in_z __format_string const wchar_t* f, ...);
//extern "C"
_Check_return_ int 
wclogprintf(__in_z __format_string const wchar_t* f, ...);
//extern "C"
_Check_return_ int 
wcerrprintf(__in_z __format_string const wchar_t* f, ...);
