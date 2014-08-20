/// http://www.codeproject.com/debug/debugout.asp?df=100&forumid=2081&exp=0&select=26147
#pragma once
#include <ostream>
#include <sstream>
#include <string>

template <class CharT, class TraitsT = std::char_traits<CharT> >
class basic_debugbuf : 
    public std::basic_stringbuf<CharT, TraitsT>
{
public:

    virtual ~basic_debugbuf()
    {
        sync();
    }

protected:

    int sync()
    {
    	std::stringbuf s;
        output_debug_string((CharT*) s.str().c_str());
        std::basic_stringbuf<CharT, TraitsT>::str(std::basic_string<CharT>());    // Clear the string buffer

        return 0;
    }

    void output_debug_string(const CharT *text) {}
};

template<class CharT, class TraitsT = std::char_traits<CharT> >
class basic_dostream : 
    public std::basic_ostream<CharT, TraitsT>
{
public:

    basic_dostream() : std::basic_ostream<CharT, TraitsT>
                (new basic_debugbuf<CharT, TraitsT>()) {}
    ~basic_dostream() 
    {
        delete std::ios::rdbuf(); 
    }
};

typedef basic_dostream<char>    dostream;
typedef basic_dostream<wchar_t> wdostream;
