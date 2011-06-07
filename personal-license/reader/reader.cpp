/**
  * Trivial mash to prevent only the most simple freetext search/replace
  * within the binary.
  */

#include "reader.h"

#include <vector>
#include <sstream>

using namespace std;

vector<unsigned char> texthex(string s)
{
    vector<unsigned char> v;
    for (unsigned i=0; i+1<s.size(); i+=2)
    {
        stringstream h(s.substr(i,2));
        int c;
        h >> hex >> c;
        v.push_back(c);
    }
    return v;
}


string backward(const vector<unsigned char>& mash)
{
    const char P0 = 173;
    const char Q = 79;
    string row2;
    char P = P0;
    for (unsigned i=0; i<mash.size(); ++i)
    {
        P ^= mash[i];
        P ^= Q;
        row2.push_back( P );
    }
    return row2;
}


READERSHARED_EXPORT string reader_text()
{
    return backward(texthex(LICENSEEMASH));
}

READERSHARED_EXPORT string reader_title()
{
    return backward(texthex(TITLEMASH));
}

