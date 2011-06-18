/**
  * Trivial mash to prevent only the most simple freetext search/replace
  * within the binary.
  */

#include "reader.h"

#include <vector>
#include <sstream>

using namespace std;

unsigned radix = 53;

#define cton_helper(low, high) \
do { \
    if (low <= c && c <= high) \
        return n + c - low; \
    n += high-low+1; \
} while (false);

unsigned cton(char c)
{
	unsigned n = 0;
	cton_helper('3', '4');
	cton_helper('6', '9');
	cton_helper('a', 'k');
	cton_helper('m', 'z');
	cton_helper('A', 'H');
	cton_helper('J', 'N');
	cton_helper('P', 'R');
	cton_helper('T', 'Y');

	cout << "Invalid character: " << (int)c << '\'' << c << '\'' << endl;
	throw string("Invalid character: ") + c;
}

std::vector<unsigned char> textradix(string s)
{
    std::vector<unsigned char> v;
	unsigned val = 0;
	unsigned byteradix = 1<<8;
	int counter = 0;

	if (s.size()%3)
		return v;

    for (unsigned i=0; i<s.size(); i++)
    {
		val *= radix;
		val += cton(s[i]);
		counter ++;

		if (counter==3)
		{
			counter = 0;
			v.push_back(val/byteradix);
			v.push_back(val%byteradix);
			val = 0;
		}
    }

    return v;
}


string backward(const std::vector<unsigned char>& mash)
{
	if (mash.size()<5)
	    return "invalid license";

	srand(mash[0] | (mash[1]<<8));
    string row2;
    char P = rand();
    if (mash[3] != (char)rand())
	    return "invalid license";
    for (unsigned i=4; i<mash.size(); ++i)
        row2.push_back( P ^= mash[i] ^ rand() );

    P ^= mash[2];
    if (P != (char)rand())
	    return "invalid license";

    return row2;
}

READERSHARED_EXPORT string reader_text()
{
    return backward(textradix(LICENSEEMASH));
}

READERSHARED_EXPORT string reader_title()
{
    return backward(textradix(TITLEMASH));
}

