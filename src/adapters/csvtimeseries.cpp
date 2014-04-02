#include "csvtimeseries.h"
#include "signal/sinksource.h"
#include "tfr/cwt.h"

#include "tasktimer.h"

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
        return OperationDesc::toString ().toStdString ();

    return QFileInfo( filename().c_str() ).fileName().toStdString();
}


QString CsvTimeseries::
        toString() const
{
    return QString::fromStdString (filename());
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
    int chunk = 1 << 18;
    Signal::pBuffer chunkBuffers;
    std::vector<float*> p;

    for (int bufferCount, channel, line=0; ifs.good();)
    {
        for (bufferCount = 0; bufferCount < chunk; ++bufferCount, ++line )
        {
            for (channel=0; ; ++channel)
            {
                if (channel >= (int)ssc.num_channels ())
                {
                    ssc = SinkSource( channel + 1 );
                    Signal::pBuffer nb(new Signal::Buffer(Signal::Interval(0, chunk), sample_rate, channel+1));
                    for (int i=0;i<channel; i++)
                        *nb->getChannel (i) |= *chunkBuffers->getChannel (i);

                    chunkBuffers = nb;

                    p.resize( channel + 1 );
                    p.back() = chunkBuffers->getChannel (channel)->waveform_data()->getCpuMemory();
                }

                ifs >> p[channel][bufferCount];

                // Eat whitespace
                while (ifs.peek () == ' ')
                    ifs.get ();

                int n = ifs.get ();
                if (n == ',' || n == ';' || n == ':' || n == '\t')
                    n = ifs.get (); // Eat delimiter if any

                if (n == '\n')
                    break; // Continue on new row
                if (n == '\r')
                {
                    if (ifs.peek() == '\n')
                        ifs.get();
                    break; // Continue on new row
                }
                if (!ifs.good())
                    break; // Couldn't read further

                ifs.putback (n);

                // Continue to read same row
            }

            // Couldn't read further
            if (!ifs.good()) {
                // Did read something on last row?
                if (channel>0) {
                    bufferCount++;
                    line++;
                }
                break;
            }
        }

        if (0 < bufferCount)
            ssc.put (BufferSource(chunkBuffers).readFixedLength (Interval(line-bufferCount,line)));
    }

    if (ssc.empty())
        throw std::ios_base::failure("Couldn't read any CSV data from '" + filename + "'");

    setBuffer( ssc.readFixedLength( ssc.getInterval() ) );

    // TODO adjust default wanted min hz to sample rate of opened signal
    //Tfr::Cwt::Singleton().set_wanted_min_hz( sample_rate/1000 );

    TaskInfo(boost::format("Signal length: %s") % lengthLongFormat());
    TaskInfo(boost::format("Data size: %lu samples, %lu channels") % number_of_samples() % num_channels() );
    TaskInfo(boost::format("Sample rate: %lu samples/second") % this->sample_rate() );
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
