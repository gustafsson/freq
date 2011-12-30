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
        :
        Signal::OperationCache(Signal::pOperation()),
        _tried_load(false),
        _sample_rate(0),
        _number_of_samples(0)
{
    _original_relative_filename = filename;
    _original_absolute_filename = QFileInfo(filename.c_str()).absoluteFilePath().toStdString();

    file.reset(new QFile(_original_absolute_filename.c_str()));
}


std::string Audiofile::
        name()
{
    if (filename().empty())
        return Operation::name();

    return QFileInfo( filename().c_str() ).fileName().toStdString();
}


Signal::IntervalType Audiofile::
        number_of_samples()
{
    tryload();

    return _number_of_samples;
}


unsigned Audiofile::
        num_channels()
{
    if ( !tryload() )
        return Signal::OperationCache::num_channels();

    return sndfile->channels();
}


float Audiofile::
        sample_rate()
{
    tryload();

    return _sample_rate;;
}


std::string Audiofile::
        filename() const
{
    return _original_relative_filename;
}


Audiofile:: // for deserialization
        Audiofile()
            :
            Signal::OperationCache(Signal::pOperation()),
            file(new QTemporaryFile()),
            _tried_load(false),
            _sample_rate(0),
            _number_of_samples(0)
{}


bool Audiofile::
        tryload()
{
    if (!sndfile)
    {
        if (_tried_load)
            return false;

        _tried_load = true;

        sndfile.reset( new SndfileHandle(file->fileName().toStdString()));

        if (0==*sndfile || 0 == sndfile->frames())
        {
            sndfile.reset();

            stringstream ss;

            ss << "Couldn't open '" << file->fileName().toStdString() << "'" << endl
               << endl
               << "Supported audio file formats through Sndfile:" << endl
               << getSupportedFileFormats();

            throw std::ios_base::failure(ss.str());
        }

        _sample_rate = sndfile->samplerate();
        _number_of_samples = sndfile->frames();

        invalidate_samples( getInterval() );
    }

    return true;
}


Signal::pBuffer Audiofile::
        readRaw( const Signal::Interval& J )

{
    if (!tryload())
    {
        TaskInfo("Loading '%s' failed (this=%p), requested %s",
                     filename().c_str(), this, J.toString().c_str());

        return zeros( J );
    }

    Signal::Interval I = J;
    Signal::IntervalType maxReadLength = 1<<18;
    if (I.count() > maxReadLength)
        I.last = I.first + maxReadLength;

    if (I.last > number_of_samples())
        I.last = number_of_samples();

    if (!I.valid())
    {
        TaskInfo("Couldn't load %s from '%s', getInterval is %s (this=%p)",
                     J.toString().c_str(), filename().c_str(), getInterval().toString().c_str(), this);
        return zeros( J );
    }

    TaskTimer tt("Loading %s from '%s' (this=%p)",
                 J.toString().c_str(), filename().c_str(), this);

    DataStorage<float> partialfile(DataStorageSize( num_channels(), I.count(), 1));
    sndfile->seek(I.first, SEEK_SET);
    sf_count_t readframes = sndfile->read(partialfile.getCpuMemory(), num_channels()*I.count()); // yes float
    if ((sf_count_t)I.count() > readframes)
        I.last = I.first + readframes;

    float* data = partialfile.getCpuMemory();

    Signal::pBuffer waveform( new Signal::Buffer(I.first, I.count(), sample_rate(), num_channels()));
    float* target = waveform->waveform_data()->getCpuMemory();

    // Compute transpose of signal
    unsigned C = waveform->channels();
    for (unsigned i=0; i<I.count(); i++) {
        for (unsigned c=0; c<C; c++) {
            target[i + c*I.count()] = data[i*C + c];
        }
    }

    tt << "Read " << I.toString() << ", total signal length " << lengthLongFormat();

    tt.flushStream();

    tt.info("Data size: %lu samples, %lu channels", (size_t)sndfile->frames(), (size_t)sndfile->channels() );
    tt.info("Sample rate: %lu samples/second", sndfile->samplerate() );

    if ((invalid_samples() - I).empty())
    {
        // Don't need this anymore
        sndfile.reset();
    }

    return waveform;
}


std::vector<char> Audiofile::
        getRawFileData(unsigned i, unsigned bytes_per_chunk)
{
    if (!file->open(QIODevice::ReadOnly))
        throw std::ios_base::failure("Couldn't get raw data from " + file->fileName().toStdString() + " (original name '" + filename() + "')");

    std::vector<char> rawFileData;

    if (bytes_per_chunk*i >= file->size())
        return rawFileData;

    file->seek(bytes_per_chunk*i);
    QByteArray bytes = file->read(bytes_per_chunk);
    file->close();

    rawFileData.resize( bytes.size() );
    memcpy(&rawFileData[0], bytes.constData(), bytes.size());

    return rawFileData;
}


void Audiofile::
        appendToTempfile( std::vector<char> rawFileData, unsigned i, unsigned bytes_per_chunk)
{
    TaskInfo ti("Audiofile::appendToTempfile(%u bytes at %u=%u*%u)", (unsigned)rawFileData.size(), i*bytes_per_chunk, i, bytes_per_chunk);

    if (rawFileData.empty())
        return;

    // file is a QTemporaryFile during deserialization
    if (!file->open(QIODevice::WriteOnly))
        throw std::ios_base::failure("Couldn't create raw data in " + file->fileName().toStdString() + " (original name '" + filename() + "')");

    file->seek(i*bytes_per_chunk);
    file->write(QByteArray::fromRawData(&rawFileData[0], rawFileData.size()));
    file->close();
}

} // namespace Adapters
