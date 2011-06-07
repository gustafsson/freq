#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>

using namespace std;

template<typename T>
string hextext(const T& s)
{
    stringstream h;
    for (unsigned i=0; i<s.size(); i++)
    {
        h << std::setfill('0') << std::setw(2) << std::hex << (int)(s[i]);
    }
    return h.str();
}

std::vector<unsigned char> texthex(string s)
{
    std::vector<unsigned char> v;
    for (unsigned i=0; i+1<s.size(); i+=2)
    {
        stringstream h(s.substr(i,2));
        int c;
        h >> std::hex >> c;
        v.push_back(c);
    }
    return v;
}

std::vector<unsigned char> forward(const string& row)
{
    std::vector<unsigned char> mash;
    const char P0 = 173;
    const char Q = 79;
    char P = P0;
    for (unsigned i=0; i<row.size(); ++i)
    {
        char R = P;
        R ^= row[i];
        R ^= Q;
        mash.push_back( R );

        P ^= R;
        P ^= Q;
    }
    return mash;
}

string backward(const std::vector<unsigned char>& mash)
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


int main(int argc, char *argv[])
{
    if (argc==2)
    {
        cout << hextext(forward(argv[1]));
        return 0;
    }

    if (argc<3)
    {
        cout << "Synopsis: " << endl;
        cout << "    mash text-to-mash" << endl;
        cout << "    mash inputfile outputfile" << endl;
        return -1;
    }

    ifstream in(argv[1]);
    ofstream out(argv[2]);
    string row;
    getline(in, row);

    cout << "Row  : " << row << endl;
    cout << "Row  : " << hextext(row) << endl;
    cout << "Row  : " << hextext(texthex(hextext(row))) << endl;

    vector<unsigned char> mash = forward(row);

    cout << "Mash : " << hextext(mash) << endl;

    string row2 = backward(mash);

    cout << "Row2 : " << hextext(row2) << endl;

    out << hextext(mash);

    return 0;
}
