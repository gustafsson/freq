#include "csvtimeseries.h"
#include "signal/sinksourcechannels.h"
#include "tfr/cwt.h"

#include <fstream>
#include <sstream>

// Qt
#include <QString>
#include <QFileInfo>
#include <QVector>
#include <QFile>
#include <QByteArray>
#include <QTemporaryFile>

using namespace std;
using namespace Signal;

namespace Adapters {

// static
std::string CsvTimeseries::
        getFileFormatsQtFilter( bool split )
{
    stringstream ss;
    if (split) {
        ss << "Comma separated values (*.csv *.txt)";
    } else {
        ss << "*.csv *.txt";
    }

    return ss.str();
}


/**
  Reads a datafile
  */
CsvTimeseries::
        CsvTimeseries(std::string filename)
{
    _original_relative_filename = filename;
    load(filename);
    rawdata = getRawFileData(filename);
}


std::string CsvTimeseries::
        name()
{
    if (filename().empty())
        return Operation::name();

    return QFileInfo( filename().c_str() ).fileName().toStdString();
}


void CsvTimeseries::
        load(std::string filename )
{
    TaskTimer tt("Loading '%s' (this=%p)", filename.c_str(), this);

    std::ifstream ifs(filename.c_str(), ios_base::in);

    float sample_rate = 1;
    //ifs >> sample_rate >> std::endl;

    SinkSourceChannels ssc;
    size_t chunk = 1 << 18;
    std::vector<Signal::pBuffer> chunkBuffers;
    std::vector<float*> p;

    for (size_t bufferCount, channel, line=0; ifs.good();)
    {
        for (bufferCount = 0; bufferCount < chunk; ++bufferCount, ++line )
        {
            for (channel=0; ; ++channel)
            {
                if (line==0)
                {
                    ssc.setNumChannels( channel + 1 );
                    chunkBuffers.resize( channel + 1 );
                    p.resize( channel + 1 );
                    chunkBuffers.back() = pBuffer( new Signal::Buffer(0, chunk, sample_rate ) );
                    p.back() = chunkBuffers.back()->waveform_data()->getCpuMemory();
                }
                else if (channel >= ssc.num_channels())
                    throw std::ios_base::failure(QString("CsvTimeseries - Unexpected format in '%1' on line %2").arg(filename.c_str()).arg(line).toStdString());

                ifs >> p[channel][bufferCount];
                int n = ifs.get();

                if (n == ',' || n == ';' || n == ':' || n == '\t')
                    continue;
                if (n == '\n')
                    break;
                if (n == '\r' && ifs.peek() == '\n')
                {
                    ifs.get();
                    break;
                }
                if (!ifs.good())
                    break;

                throw std::ios_base::failure(QString("CsvTimeseries - Unexpected format in '%1' on line %2").arg(filename.c_str()).arg(line).toStdString());
            }

            if (ifs.good() && channel + 1 != ssc.num_channels())
                throw std::ios_base::failure(QString("CsvTimeseries - Unexpected format in '%1' on line %2").arg(filename.c_str()).arg(line).toStdString());

            if (!ifs.good())
                break;
        }

        if (0 < bufferCount) for (channel=0; channel < ssc.num_channels(); ++channel)
        {
            chunkBuffers[channel]->sample_offset = (double)(line - bufferCount);
            ssc.set_channel( channel );
            ssc.put( BufferSource( chunkBuffers[channel] ).readFixedLength( Interval( line - bufferCount, line )) );
        }

    }

    _waveforms.resize( ssc.num_channels());
    for (unsigned c=0; c<ssc.num_channels(); c++)
    {
        ssc.set_channel( c );
        _waveforms[c] = ssc.readFixedLength( ssc.getInterval() );
    }
    Tfr::Cwt::Singleton().set_wanted_min_hz( sample_rate/1000 );

    tt << "Signal length: " << lengthLongFormat();

    tt.flushStream();

    tt.info("Data size: %lu samples, %lu channels", number_of_samples(), num_channels() );
    tt.info("Sample rate: %lu samples/second", this->sample_rate() );
}


std::vector<char> CsvTimeseries::
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


void CsvTimeseries::
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