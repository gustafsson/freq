#ifdef _MSC_VER
typedef __int64 __int64_t;
#else
#include <stdint.h> // defines __int64_t which is expected by sndfile.h
#endif

#include "audiofile.h"
#include "Statistics.h" // to play around for debugging

#include <sndfile.hh> // for reading various formats
#include <math.h>
#include <stdexcept>
#include <iostream>
#include <sstream>

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/algorithm/string.hpp>

#if LEKA_FFT
#include <cufft.h>
#endif

// Qt
#include <QFileInfo>
#include <QVector>
#include <QFile>
#include <QByteArray>
#include <QTemporaryFile>

using namespace std;

namespace Adapters {

static std::string getSupportedFileFormats (bool detailed=false) {
    SF_FORMAT_INFO	info ;
    SF_INFO 		sfinfo ;
    char buffer [128] ;
    int format, major_count, subtype_count, m, s ;
    stringstream ss;

    memset (&sfinfo, 0, sizeof (sfinfo)) ;
    buffer [0] = 0 ;
    sf_command (NULL, SFC_GET_LIB_VERSION, buffer, sizeof (buffer)) ;
    if (strlen (buffer) < 1)
    {	ss << "Could not retrieve sndfile lib version.";
        return ss.str();
    }
    ss << "Version : " << buffer << endl;

    sf_command (NULL, SFC_GET_FORMAT_MAJOR_COUNT, &major_count, sizeof (int)) ;
    sf_command (NULL, SFC_GET_FORMAT_SUBTYPE_COUNT, &subtype_count, sizeof (int)) ;

    sfinfo.channels = 1 ;
    for (m = 0 ; m < major_count ; m++)
    {	info.format = m ;
            sf_command (NULL, SFC_GET_FORMAT_MAJOR, &info, sizeof (info)) ;
            ss << info.name << "  (extension \"" << info.extension << "\")" << endl;

            format = info.format ;

            if(detailed)
            {
                for (s = 0 ; s < subtype_count ; s++)
                {	info.format = s ;
                        sf_command (NULL, SFC_GET_FORMAT_SUBTYPE, &info, sizeof (info)) ;

                        format = (format & SF_FORMAT_TYPEMASK) | info.format ;

                        sfinfo.format = format ;
                        if (sf_format_check (&sfinfo))
                                ss << "   " << info.name << endl;
                } ;
                ss << endl;
            }
    } ;
    ss << endl;

    return ss.str();
}


// static
std::string Audiofile::
        getFileFormatsQtFilter( bool split )
{
    SF_FORMAT_INFO	info ;
    SF_INFO 		sfinfo ;
    char buffer [128] ;

	int major_count, subtype_count, m ;
    stringstream ss;

    memset (&sfinfo, 0, sizeof (sfinfo)) ;
    buffer [0] = 0 ;
    sf_command (NULL, SFC_GET_LIB_VERSION, buffer, sizeof (buffer)) ;
    if (strlen (buffer) < 1)
    {
        return NULL;
    }

    sf_command (NULL, SFC_GET_FORMAT_MAJOR_COUNT, &major_count, sizeof (int)) ;
    sf_command (NULL, SFC_GET_FORMAT_SUBTYPE_COUNT, &subtype_count, sizeof (int)) ;

    sfinfo.channels = 1 ;
    for (m = 0 ; m < major_count ; m++)
    {	info.format = m ;
            sf_command (NULL, SFC_GET_FORMAT_MAJOR, &info, sizeof (info)) ;
            if (split) {
                if (0<m) ss << ";;";
                string name = info.name;
                boost::replace_all(name, "(", "- ");
                boost::erase_all(name, ")");
                ss << name << " (*." << info.extension << " *." << info.name << ")";
            } else {
                if (0<m) ss << " ";
                ss <<"*."<< info.extension << " *." << info.name;
            }
    }

    return ss.str();
}


/**
  Reads an audio file using libsndfile
  */
Audiofile::
        Audiofile(std::string filename)
{
    _original_relative_filename = filename;
    load(filename);
    rawdata = getRawFileData(filename);
}


std::string Audiofile::
        name()
{
    if (filename().empty())
        return Operation::name();

    return QFileInfo( filename().c_str() ).fileName().toStdString();
}


void Audiofile::
        load(std::string filename )
{
    TaskTimer tt("Loading '%s' (this=%p)", filename.c_str(), this);

    SndfileHandle source(filename);

    if (0==source || 0 == source.frames()) {
        stringstream ss;

        ss << "Couldn't open '" << filename << "'" << endl
           << endl
           << "Supported audio file formats through Sndfile:" << endl
           << getSupportedFileFormats();

        throw std::ios_base::failure(ss.str());
    }

    GpuCpuData<float> completeFile(0, make_uint3( source.channels(), source.frames(), 1));
    source.read(completeFile.getCpuMemory(), source.channels()*source.frames()); // yes float
    float* data = completeFile.getCpuMemory();

	Signal::pBuffer waveform( new Signal::Buffer(0, source.frames(), source.samplerate(), source.channels()));
    float* target = waveform->waveform_data()->getCpuMemory();

    // Compute transpose of signal
    for (unsigned i=0; i<source.frames(); i++) {
        for (int c=0; c<source.channels(); c++) {
            target[i + c*source.frames()] = data[i*source.channels() + c];
        }
    }

	setBuffer( waveform );

    //_waveform = getChunk( 0, number_of_samples(), 0, Waveform_chunk::Only_Real );
    //Statistics<float> waveform( _waveform->waveform_data.get() );

    tt << "Signal length: " << lengthLongFormat();

    tt.flushStream();

    tt.info("Data size: %lu samples, %lu channels", (size_t)source.frames(), (size_t)source.channels() );
    tt.info("Sample rate: %lu samples/second", source.samplerate() );
}


std::vector<char> Audiofile::
        getRawFileData(std::string filename)
{
    QFile f(QString::fromLocal8Bit( filename.c_str() ));
    if (!f.open(QIODevice::ReadOnly))
        throw std::ios_base::failure("Couldn't get raw data from " + filename);

    QByteArray bytes = f.readAll();
    std::vector<char>rawFileData;
    rawFileData.resize( bytes.size() );
    memcpy(&rawFileData[0], bytes.constData(), bytes.size());

    return rawFileData;
}


void Audiofile::
        load( std::vector<char>rawFileData)
{
    TaskInfo ti("Audiofile::load(rawFile)");

    QTemporaryFile file;
    if (!file.open())
        throw std::ios_base::failure("Couldn't create raw data");

    std::string filename = file.fileName().toStdString();
    file.write(QByteArray::fromRawData(&rawFileData[0], rawFileData.size()));
    file.close();

    load(filename);
}

} // namespace Adapters
