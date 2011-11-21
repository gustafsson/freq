/**
  * Trivial mash to prevent only the most simple freetext search/replace
  * within the binary.
  */

#include "reader.h"

#include <vector>
#include <sstream>
#include <QSettings>
#include "enterlicense.h"

#include <QApplication>
#include <stdexcept>
#include <TaskTimer.h>
#include <QDate>
#include <QTextStream>

using namespace std;

namespace Sawe
{

std::string Reader::name;

unsigned radix = 52;

#define cton_helper(low, high) \
do { \
    if (low <= c && c <= high) \
        return n + c - low; \
    n += high-low+1; \
} while (false);

static inline unsigned cton(char c)
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

    throw invalid_argument(string("Invalid character: ") + c);
}


static inline vector<unsigned char> textradix(string s)
{
    vector<unsigned char> v;
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


unsigned long long X = 1;
#ifndef _MSC_VER
#define A 8433437992146984169LLU
#define B 7905438737954111703LLU
#else
#define A 8433437992146984169
#define B 7905438737954111703
#endif
static inline void pseudoseed(unsigned long long seed)
{
    X = seed;
}
static inline unsigned char pseudorand()
{
    X = A*X + B;
    return X / (2^56);
}

static inline string backward(const std::vector<unsigned char>& mash)
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

inline vector<unsigned char> forward2(const string& row)
{
    // nothing is encrypted, the mashed key is just obfuscated by a "secret algorithm"
    vector<unsigned char> mash;
    pseudoseed( time(NULL) );
    unsigned short s = pseudorand()^(pseudorand() << 8);
    mash.push_back( s );
    mash.push_back( s>>8 );
    mash.push_back( 0 );
    pseudoseed( s );
    for (unsigned i=0; i<row.size(); ++i)
    {
        mash.push_back( pseudorand() ^ row[i] );
    }
    mash[2] = row.size() ^ pseudorand();

    return mash;
}

static inline string backward2(const vector<unsigned char>& mash)
{
    // nothing is encrypted, the mashed key is just obfuscated by a "secret algorithm"
    if (mash.size()<5)
        return "";

    pseudoseed(mash[0] | (mash[1]<<8));
    string row2;
    for (unsigned i=3; i<mash.size(); ++i)
        row2.push_back( pseudorand() ^ mash[i] );

    if (mash[2] != (row2.size() ^ pseudorand()))
        return "";

    return row2;
}


vector<unsigned char> Reader::
        mash(const string& unmashed)
{
    return forward2(unmashed);
}


string Reader::
        unmash(const vector<unsigned char>& mash)
{
    return backward2(mash);
}


string Reader::
        tryread(string mash)
{
    try
    {
        for(static bool once = true; once; once=false) TaskInfo("found %s", mash.c_str());

        string lic = backward(textradix(mash));
        QString qlic = QString::fromStdString(lic);
        QStringList parts = qlic.split("|");
        if (parts.size()<3)
            return "";

        QString type = parts[0];
        QString licensee = parts[1];
        Reader::name = licensee.toStdString();
        QString expires = parts[2];
        QTextStream qts(&expires);
        int year=-1, month=-1, day=-1;
        char dummy=-1;
        qts >> year >> dummy >> month >> dummy >> day;
        if (-1==day)
            return "";

        if (0==year)
        {
            QString licenseText;

            if (type!="-")
            {
                licenseText = type + " of ";
            }

            licenseText+="Sonic AWE licensed to " + licensee;
            return licenseText.toStdString();
        }
        else
        {
            QDate qd(year, month, day);
            if (qd.addMonths(1) < QDate::currentDate())
                return "";

            if (qd.addMonths(1) < QDate::fromString(QString(__DATE__).replace("  ", " "), "MMM d yyyy"))
                return "";

            QString licenseText;

            if (type!="-")
            {
                licenseText = type + " of ";
            }

            licenseText+="Sonic AWE licensed to " + licensee + " until " + expires;
            return licenseText.toStdString();
        }
    }
    catch (invalid_argument x)
    {
        TaskInfo("entered %s", mash.c_str());
    }
    return "";
}

string Reader::
        reader_text(bool annoy)
{
#if defined(TARGET_reader)
    return "Sonic AWE Reader";
#else
    while (true)
    {
        if (QSettings().contains("value"))
        {
            string LICENSEEMASH = QSettings().value("value").toString().toStdString();
            // try to parse
            string lic = tryread(LICENSEEMASH);
            if (!lic.empty())
                return lic;
        }

        if (!annoy)
            return "not licensed";

        Sawe::EnterLicense lic;
        if (QDialog::Accepted == lic.exec())
        {
            QString LICENSEEMASH = lic.lineEdit();
            QSettings().setValue("value", LICENSEEMASH);
        }
        else
        {
            return "not licensed";
        }
    }
#endif
}

string Reader::
        reader_title()
{
    return reader_text();
}

} // namespace Sawe
