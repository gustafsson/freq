#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <stdlib.h>

#ifdef _MSC_VER
#include <time.h>
#endif

using namespace std;

unsigned long long X = 1;
unsigned long long A = 8433437992146984169;
unsigned long long B = 7905438737954111703;
void pseudoseed(unsigned long long seed)
{
	X = seed;
}
unsigned char pseudorand()
{
	X = A*X + B;
	return X / (2^56);
}


unsigned radix = 52; // >= 41 is enough for 3 letters for 2 bytes

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
	cton_helper('A', 'C');
	cton_helper('E', 'H');
	cton_helper('J', 'N');
	cton_helper('P', 'R');
	cton_helper('T', 'Y');

	cout << "Invalid character: " << (int)c << '\'' << c << '\'' << endl;
	throw string("Invalid character: ") + c;
}

#define ntoc_helper(low, high) \
do { \
	unsigned l = high-low+1; \
	if (n<l) \
	    return low + n; \
	n -= l; \
} while (false);

char ntoc(unsigned n)
{
	ntoc_helper('3', '4');
	ntoc_helper('6', '9');
	ntoc_helper('a', 'k');
	ntoc_helper('m', 'z');
	ntoc_helper('A', 'C');
	ntoc_helper('E', 'H');
	ntoc_helper('J', 'N');
	ntoc_helper('P', 'R');
	ntoc_helper('T', 'Y');

	cout << "Invalid number: " << n + radix << endl;
	throw string("Invalid number");
}

template<typename T>
string radixtext(const T& s)
{
    stringstream h;
	unsigned val = 0;
	unsigned counter=0;
	unsigned byteradix = 1<<8;

	if (s.size()%2)
	{
	    for (unsigned i=0; i<s.size(); i++)
			cout << "s[" << i << "]=" << (int)s[i] << endl;
		return "must be multiple of 2 in length";
	}

    for (unsigned i=0; i<s.size(); i++)
    {
		val *= byteradix;
		val += s[i];
		counter ++;

		if (counter==2)
		{
			counter = 0;
			h << ntoc(val/radix/radix);
			h << ntoc((val/radix)%radix);
			h << ntoc(val%radix);
			val=0;
		}
    }

    return h.str();
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

std::vector<unsigned char> forward(string row)
{
    // nothing is encrypted, the mashed key is just obfuscated by a "secret algorithm"
	if (row.size()%2)
		row.push_back(0);

    std::vector<unsigned char> mash;
	pseudoseed( time(NULL) );
	unsigned short s = pseudorand()^(pseudorand() << 8);
    mash.push_back( s );
    mash.push_back( s>>8 );
	pseudoseed( s );
    unsigned char P = pseudorand();
    mash.push_back( 0 );
    mash.push_back( pseudorand() );
	for (unsigned i=0; i<row.size(); ++i)
    {
        mash.push_back( P ^ pseudorand() ^ row[i] );

        P = row[i];
    }
    mash[2]=( P ^ pseudorand() );

    return mash;
}

string backward(const std::vector<unsigned char>& mash)
{
    // nothing is encrypted, the mashed key is just obfuscated by a "secret algorithm"
	if (mash.size()<5)
	    return "invalid license";

	pseudoseed(mash[0] | (mash[1]<<8));
    string row2;
    unsigned char P = pseudorand();
    if (mash[3] != pseudorand())
	    return "invalid license";
    for (unsigned i=4; i<mash.size(); ++i)
        row2.push_back( P ^= mash[i] ^ pseudorand() );

    P ^= mash[2];
    if (P != pseudorand())
	    return "invalid license";

    if (0 == row2[row2.length()-1])
        row2 = row2.substr(0, row2.length()-1);

    return row2;
}


int main(int argc, char *argv[])
{
    if (argc==2)
    {
        cout << radixtext(forward(argv[1]));
        return 0;
    }

    if (argc<3)
    {
        cout << "Synopsis: " << endl;
        cout << "    mash text-to-mash" << endl;
        cout << "    mash inputfile outputfile" << endl;
        return -1;
    }

	try
	{
		ifstream in(argv[1]);
		ofstream out(argv[2]);
		string row;
		getline(in, row);

		if (row.size()%2)
			row.push_back('\0');

		cout << "Row  : " << row << endl;
		cout << "Row  : " << radixtext(row) << endl;
		cout << "Row  : " << radixtext(textradix(radixtext(row))) << endl;

		vector<unsigned char> mash = forward(row);

		cout << "Mash : " << radixtext(mash) << endl;

		string row2 = backward(mash);
		if (row2.size()%2)
			row2.push_back('\0');

		cout << "Row2 : " << radixtext(row2) << endl;
		cout << "Row2 : " << row2 << endl;

		out << radixtext(mash);

		return 0;
	}
	catch (const string& s)
	{
		cout << "Caught exception: " << s << endl;
	}
}

