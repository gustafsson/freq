#pragma once

#include <istream>
#include <ostream>

template<typename t>
class redirectStream
{
public:
	redirectStream( std::basic_ostream <t>* o,  std::basic_streambuf<t>* newBuf )
	:	i(0),
		o(o),
		orig(o->rdbuf( newBuf ))
	{}
	redirectStream( std::basic_istream <t>* i,  std::basic_streambuf<t>* newBuf )
	:	i(i),
		o(0),
		orig(i->rdbuf( newBuf ))
	{}
	~redirectStream()
	{
		if(i) i->rdbuf(orig);
		if(o) o->rdbuf(orig);
	}
private:
	std::basic_istream<t>* i;
	std::basic_ostream<t>* o;
	std::basic_streambuf<t>* orig;
};
