#include "stringprintf.h"

#include <stdexcept>
#include <memory>

#include <boost/shared_array.hpp>

#include "cva_list.h"
#include <iostream>
#include <sstream>

using namespace std;
using namespace boost;

//fva_list cheatlist;

template<> std::basic_string<char>
printfstring(const char* f, va_list argList, int* result)
{
	std::basic_string<char> s;

	int
//#ifdef _MSC_VER
                //c = vsnprintf_s( 0, 0, 0, f, Cva_list(argList) );
//#else
		c = vsnprintf( 0, 0, f, Cva_list(argList) );
//#endif
        if( 0 > c ) {
                std::basic_stringstream<char> ss;
                ss << "printfstring<char> could not parse format string. " << endl
                   << "0 > c(" << c << ")" << endl
                   << "f = " << f;
                cout << ss.str() << endl;
                throw std::invalid_argument(ss.str());
            }

        shared_array<char> t( new char[c+1] );
#ifdef _MSC_VER
        int r =  vsnprintf_s( t.get(), c+1, c, f, Cva_list(argList) );
#else
        int r =  vsnprintf( t.get(), c+1, f, Cva_list(argList) );
#endif
        if( 0 != result)
                *result = r;

        if( c != r ) {
                std::basic_stringstream<char> ss;
                ss << "printfstring<char> could not write result. " << endl
                        << "c(" << c << ") != r(" << r << ")" << endl
                        << "f = " << f << endl;
                if (t[c]) {
                    ss << " t[" << c << "] = " << t[c] << endl;
                    t[c] = 0;
                }
                ss << "t = " << t.get();
                cout << ss.str() << endl;
                throw std::invalid_argument( ss.str() );
            }
	s.append( t.get(), c );

	return s;
}

/*template<> std::basic_string<wchar_t>
printfstring( const wchar_t* f, va_list argList, int* result)
{
	std::basic_string<wchar_t> s;
	
	int
	  c = vsnprintf( 0, 0, f, argList );
	if( 0 > c )
		throw std::invalid_argument("printfstring<wchar_t> could not parse format string");
	shared_array<wchar_t> t( new wchar_t[c+1] );
	int r = vsnprintf( t.get(), c+1, f, argList );
	if( 0 != result )
		*result = r;
	if( c != r )
		throw std::invalid_argument("printfstring<wchar_t> could not write result");
	s.append( t.get(), c );

	return s;
	}*/

template<> std::basic_string<char>
printfstring<char>( const char* f, ...)
{
	int discardedResult;
        Cva_start( c, f );
        return printfstring(f, c, &discardedResult);
}

/*template<> std::basic_string<wchar_t>
printfstring<wchar_t>( const wchar_t* f, ...)
{
	int discardedResult;
	return printfstring(f, cva_list( f ), &discardedResult);
	}*/

template<> int
stringprintf<char>(std::basic_string<char>& s, const char* f, ...)
{
	int r=-1;
        Cva_start( c, f );
        s += printfstring(f, c, &r );
	return r;
}

/*template<> int
stringprintf<wchar_t>(std::basic_string<wchar_t>& s, const wchar_t* f, ...)
{
	int r=-1;
	s += printfstring(f, cva_list( f ), &r );
	return r;
	}*/

template<> int
ostreamprintf<char>(std::basic_ostream<char>& o, const char* f, va_list args)
{
	int r=-1;
	std::basic_string<char> s = printfstring(f, args, &r);
	o << s.c_str();
	return r;
}
/*template<> int
ostreamprintf<wchar_t>(std::basic_ostream<wchar_t>& o, const wchar_t* f, va_list args)
{
	int r = -1;
	std::basic_string<wchar_t> s = printfstring(f, args, &r);
	o << s.c_str();
	return r;
	}*/

template<> int
ostreamprintf<char>(std::basic_ostream<char>& o, const char* f, ...)
{
        Cva_start( c, f );
        return ostreamprintf(o, f, c);
}
/*template<> int
ostreamprintf<wchar_t>(std::basic_ostream<wchar_t>& o, const wchar_t* f, ...)
{
	return ostreamprintf(o, f, cva_list(f));
	}*/


int coutprintf(const char* f, ...){
        Cva_start( c, f );
        return ostreamprintf(std::cout, f, c);
}
int clogprintf(const char* f, ...){
        Cva_start( c, f );
        return ostreamprintf(std::clog, f, c);
}
int cerrprintf(const char* f, ...){
        Cva_start( c, f );
        return ostreamprintf(std::cerr, f, c);
}
/*int wcoutprintf(const wchar_t* f, ...){
        Cva_start( c, f );
        return ostreamprintf(std::wcout, f, c);
}
int wclogprintf(const wchar_t* f, ...){
        Cva_start( c, f );
        return ostreamprintf(std::wclog, f, c);
}
int wcerrprintf(const wchar_t* f, ...){
        Cva_start( c, f );
        return ostreamprintf(std::wcerr, f, c);
	}*/
