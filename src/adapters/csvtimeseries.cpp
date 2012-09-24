#include "csvtimeseries.h"
#include "signal/sinksource.h"
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


// static
bool CsvTimeseries::
        hasExpectedSuffix( const std::string& suffix )
{
    return "csv" == suffix;
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
    if (!ifs.is_open())
        throw std::ios_base::failure("Couldn't open file: " + filename);

    float sample_rate = 1;
    //ifs >> sample_rate >> std::endl;

    SinkSource ssc(0);
    size_t chunk = 1 << 18;
    std::vector<Signal::pMonoBuffer> chunkBuffers;
    std::vector<float*> p;

    for (size_t bufferCount, channel, line=0; ifs.good();)
    {
        for (bufferCount = 0; bufferCount < chunk; ++bufferCount, ++line )
        {
            for (channel=0; ; ++channel)
            {
                if (line==0)
                {
                    ssc = SinkSource( channel + 1 );
                    chunkBuffers.resize( channel + 1 );
                    p.resize( channel + 1 );
                    chunkBuffers.back() = pMonoBuffer( new MonoBuffer(0, chunk, sample_rate ) );
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
            pMonoBuffer mb( new MonoBuffer((double)(line - bufferCount), chunkBuffers[channel]->waveform_data(), sample_rate));
            pBuffer b( new Buffer(mb));
            ssc.put( BufferSource( b ).readFixedLength( Interval( line - bufferCount, line )) );
        }

    }

    if (ssc.empty())
        throw std::ios_base::failure("Couldn't read any CSV data from '" + filename + "'");

    setBuffer( ssc.readFixedLength( ssc.getInterval() ) );

    // TODO adjust default wanted min hz to sample rate of opened signal
    //Tfr::Cwt::Singleton().set_wanted_min_hz( sample_rate/1000 );

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
