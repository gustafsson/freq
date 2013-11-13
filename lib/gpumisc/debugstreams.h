// Define the macro REDIRECT_COUT_TO_DEBUG to make 
// cout and wcout redirect to debug output

#include <iostream>
#include <memory> // auto_ptr

#include "debugbuf.h"
#include "redirectStream.h"

class DebugStreams
{
public:
	DebugStreams( bool standardOutput = false)
	{
		errRedir.reset( new redirectStream<char>( &std::cerr, &d) );
		werrRedir.reset( new redirectStream<wchar_t>( &std::wcerr, &wd) );
		if ( standardOutput ) {
			outRedir.reset( new redirectStream<char>( &std::cout, &d) );
			woutRedir.reset( new redirectStream<wchar_t>( &std::wcout, &wd) );
		}
	}
	~DebugStreams()
	{
		errRedir.reset();
		werrRedir.reset();
		outRedir.reset();
		woutRedir.reset();
	}
private:
	// use auto_ptr to make sure redirectStream is destroyed before basic_debugbuf
	std::auto_ptr<redirectStream<char> > errRedir;
	std::auto_ptr<redirectStream<wchar_t> > werrRedir;
	std::auto_ptr<redirectStream<char> > outRedir;
	std::auto_ptr<redirectStream<wchar_t> > woutRedir;

	basic_debugbuf<char> d;
	basic_debugbuf<wchar_t> wd;
	basic_debugbuf<char> o;
	basic_debugbuf<wchar_t> wo;
};

#ifdef REDIRECT_COUT_TO_DEBUG
std::auto_ptr<DebugStreams> useDebugoutput( new DebugStreams(REDIRECT_COUT_TO_DEBUG) );
#endif