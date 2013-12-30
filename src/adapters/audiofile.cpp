#include "audiofile.h"
#include "Statistics.h" // to play around for debugging
#include "signal/transpose.h"
#include "neat_math.h" // defines __int64_t which is expected by sndfile.h

#include <sndfile.hh> // for reading various formats
#include <math.h>
#include <stdexcept>
#include <iostream>
#include <sstream>

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/format.hpp>

#if LEKA_FFT
#include <cufft.h>
#endif

// Qt
#include <QFileInfo>
#include <QVector>
#include <QFile>
#include <QByteArray>
#include <QTemporaryFile>


//#define VERBOSE_AUDIOFILE
#define VERBOSE_AUDIOFILE if(0)

//#define TIME_AUDIOFILE_LINE(x) TIME(x)
#define TIME_AUDIOFILE_LINE(x) x


using namespace std;
using namespace boost;

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
            //ss << info.name << "  (extension \"" << info.extension << "\")" << endl;
            ss << info.name;
            if (m+2 < major_count)
                ss << ", ";
            else if ( m+2 == major_count)
                ss << " and ";

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


// static
bool Audiofile::
        hasExpectedSuffix( const std::string& suffix )
{
    SF_FORMAT_INFO	info;
    SF_INFO 		sfinfo;
    char            buffer[128];

    int major_count, subtype_count, m;

    memset( &sfinfo, 0, sizeof (sfinfo) );
    buffer [0] = 0;
    sf_command( NULL, SFC_GET_LIB_VERSION, buffer, sizeof (buffer) );
    if (strlen(buffer) < 1)
    {
        return false;
    }

    sf_command( NULL, SFC_GET_FORMAT_MAJOR_COUNT, &major_count, sizeof (int) );
    sf_command( NULL, SFC_GET_FORMAT_SUBTYPE_COUNT, &subtype_count, sizeof (int) );

    sfinfo.channels = 1;
    for (m = 0 ; m < major_count ; m++)
    {
        info.format = m;
        sf_command( NULL, SFC_GET_FORMAT_MAJOR, &info, sizeof (info));

        std::string extension = info.extension;
        for (unsigned i=0; i<extension.size(); ++i)
            extension[i] = std::tolower(extension[i]);

        if (extension == suffix)
            return true;
    }

    return false;
}


/**
  Reads an audio file using libsndfile
  */
Audiofile::
        Audiofile(std::string filename)
        :
        _tried_load(false),
        _sample_rate(0),
        _number_of_samples(0),
        _number_of_channels(0)
{
    _original_relative_filename = filename;
    _original_absolute_filename = QFileInfo(filename.c_str()).absoluteFilePath().toStdString();

    file.reset(new QFile(_original_absolute_filename.c_str()));

    // Read the header and throw an exception if it can't be read
    tryload();
}


std::string Audiofile::
        name()
{
    if (filename().empty())
        return vartype(*this);

    return QFileInfo( filename().c_str() ).fileName().toStdString();
}


Signal::IntervalType Audiofile::
        number_of_samples()
{
    if (0==_number_of_samples)
        tryload();

    return _number_of_samples;
}


unsigned Audiofile::
        num_channels()
{
    if (0==_number_of_channels)
        tryload();

    return std::max(1u, _number_of_channels);
}


float Audiofile::
        sample_rate()
{
    if (0==_sample_rate)
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
            file(new QTemporaryFile()),
            _tried_load(false),
            _sample_rate(0),
            _number_of_samples(0),
            _number_of_channels(0)
{
}


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
        _number_of_channels = sndfile->channels();
    }

    return true;
}


Signal::pBuffer Audiofile::
        readRaw( const Signal::Interval& J )
{
    Signal::Interval I = readRawInterval(J);

    if (!(I & getInterval ()))
        return zeros(I);

    // Need the sound file to be available to read data.
    EXCEPTION_ASSERTX(tryload(), str(format("Loading '%s' failed (this=%p), requested %s") %
                                        filename() % this % J.toString()));

    boost::shared_ptr<TaskTimer> tt;
    VERBOSE_AUDIOFILE tt.reset(new TaskTimer("Loading %s from '%s' (this=%p)",
                 I.toString().c_str(), filename().c_str(), this));

    DataStorage<float> partialfile(DataStorageSize( num_channels(), I.count(), 1));
    sf_count_t sndfilepos;
    TIME_AUDIOFILE_LINE( sndfilepos = sndfile->seek(I.first, SEEK_SET) );
    if (sndfilepos < 0)
    {
        TaskInfo("%s", str(format("ERROR! Couldn't set read position to %d. An error occured (%d)") % I.first % sndfilepos).c_str());
        return zeros( J );
    }
    if (sndfilepos != I.first)
    {
        TaskInfo("%s", str(format("ERROR! Couldn't set read position to %d. sndfilepos was %d") % I.first % sndfilepos).c_str());
        return zeros( J );
    }

    float* data = CpuMemoryStorage::WriteAll<float,3>( &partialfile ).ptr();
    sf_count_t readframes;
    TIME_AUDIOFILE_LINE( readframes = sndfile->read(data, num_channels()*I.count())); // read float
    if ((sf_count_t)I.count() > readframes)
        I.last = I.first + readframes;

    Signal::pBuffer waveform( new Signal::Buffer(I.first, I.count(), sample_rate(), num_channels()));
    Signal::pTimeSeriesData mergedata = waveform->mergeChannelData ();
    TIME_AUDIOFILE_LINE( Signal::transpose( mergedata.get(), &partialfile ) );
    waveform.reset (new Signal::Buffer(I.first, mergedata, sample_rate()));

    VERBOSE_AUDIOFILE *tt << "Read " << I.toString() << ", total signal length " << lengthLongFormat();

    VERBOSE_AUDIOFILE tt->flushStream();

    VERBOSE_AUDIOFILE tt->info("Data size: %lu samples, %lu channels", (size_t)sndfile->frames(), (size_t)sndfile->channels() );
    VERBOSE_AUDIOFILE tt->info("Sample rate: %lu samples/second", sndfile->samplerate() );

    return waveform;
}


Signal::Interval Audiofile::
        readRawInterval( const Signal::Interval& J )
{
    Signal::Interval L = getInterval ();
    Signal::Interval I;
    Signal::IntervalType fixedReadLength = 1<<20;

    I.first = align_down(J.first,fixedReadLength);
    I.last = I.first + fixedReadLength;

    if (J.first < L.first) {
        // Treat out of range samples as zeros.
        I.last = std::min(I.last, L.first);
    } else if (J.first < L.last) {
        I &= L;
    } else {
        // Treat out of range samples as zeros.
        I.first = std::max(I.first, L.last);
    }

    return  I;
}


class CloseAfterScope
{
public:
    CloseAfterScope(boost::weak_ptr<QFile> file):file(file) {}
    ~CloseAfterScope() {
        boost::shared_ptr<QFile> filep = file.lock();
        if(filep && filep->isOpen()) filep->close();
    }
private:
    boost::weak_ptr<QFile> file;
};

std::vector<char> Audiofile::
        getRawFileData(unsigned i, unsigned bytes_per_chunk)
{
    TaskInfo ti("Audiofile::getRawFileData(at %u=%u*%u from %s)",
                bytes_per_chunk*i, i, bytes_per_chunk,
                file->fileName().toStdString().c_str());

    CloseAfterScope cas(file);
    if (!file->open(QIODevice::ReadOnly))
        throw std::ios_base::failure("Couldn't get raw data from " + file->fileName().toStdString() + " (original name '" + filename() + "')");

    std::vector<char> rawFileData;

    if (bytes_per_chunk*i >= file->size())
        return rawFileData;

    file->seek(bytes_per_chunk*i);
    QByteArray bytes = file->read(bytes_per_chunk);

    rawFileData.resize( bytes.size() );
    memcpy(&rawFileData[0], bytes.constData(), bytes.size());

    return rawFileData;
}


void Audiofile::
        appendToTempfile(std::vector<char> rawFileData, unsigned i, unsigned bytes_per_chunk)
{
    TaskInfo ti("Audiofile::appendToTempfile(%u bytes at %u=%u*%u)",
                (unsigned)rawFileData.size(), i*bytes_per_chunk, i, bytes_per_chunk);

    if (rawFileData.empty())
        return;

    // file is a QTemporaryFile during deserialization
    CloseAfterScope cas(file);

    if (!file->open(QIODevice::WriteOnly))
        throw std::ios_base::failure("Couldn't create raw data in " + file->fileName().toStdString() + " (original name '" + filename() + "')");

    file->seek(i*bytes_per_chunk);
    file->write(QByteArray::fromRawData(&rawFileData[0], rawFileData.size()));
}


AudiofileOperation::
        AudiofileOperation(Audiofile::Ptr audiofile) : audiofile_(audiofile)
{}


Signal::pBuffer AudiofileOperation::
        process(Signal::pBuffer b)
{
    Signal::pBuffer p = audiofile_->readRaw(b->getInterval ());
    EXCEPTION_ASSERT_EQUALS(p->getInterval (), b->getInterval ());
    return p;
}


AudiofileDesc::
        AudiofileDesc(boost::shared_ptr<Audiofile> audiofile)
    :
      audiofile_(audiofile)
{}


Signal::Interval AudiofileDesc::
        requiredInterval( const Signal::Interval& J, Signal::Interval* expectedOutput ) const
{
    Signal::Interval I = audiofile_->readRawInterval (J);

    if (expectedOutput)
        *expectedOutput = I;

    return I;
}


Signal::Interval AudiofileDesc::
        affectedInterval( const Signal::Interval& I ) const
{
    return I;
}


Signal::Operation::Ptr AudiofileDesc::
        createOperation(Signal::ComputingEngine* engine) const
{
    if (0 == engine)
        return Signal::Operation::Ptr(new AudiofileOperation(audiofile_));

    return Signal::Operation::Ptr();
}


Signal::OperationDesc::Ptr AudiofileDesc::
        copy() const
{
    return OperationDesc::Ptr(new AudiofileDesc(audiofile_));
}


Signal::OperationDesc::Extent AudiofileDesc::
        extent() const
{
    Extent x;
    x.interval = audiofile_->getInterval ();
    x.number_of_channels = audiofile_->num_channels ();
    x.sample_rate = audiofile_->sample_rate ();
    return x;
}


QString AudiofileDesc::
        toString() const
{
    return audiofile_->filename ().c_str ();
}


bool AudiofileDesc::
        operator==(const OperationDesc& d) const
{
    if (const AudiofileDesc* a = dynamic_cast<const AudiofileDesc*>(&d))
        return a->audiofile_ == this->audiofile_;
    return false;
}

} // namespace Adapters
