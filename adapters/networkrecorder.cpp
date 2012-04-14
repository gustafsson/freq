#include "networkrecorder.h"

#include <QMutexLocker>
#include <QTcpSocket>

namespace Adapters {

NetworkRecorder::NetworkRecorder(QUrl url, float samplerate)
:
    url(url),
    samplerate(samplerate)
{
    if (url.scheme() != "s16bursts")
    {
        // TODO display error message
        // only our own special newly invented scheme 's16bursts' is supported
    }

    connect(&tcpSocket, SIGNAL(readyRead()), this, SLOT(readData()));
}


NetworkRecorder::~NetworkRecorder()
{
}


void NetworkRecorder::
        startRecording()
{
    _offset = length();

    // TODO connect to url and call 'receivedData' asynchronously when data is received
    tcpSocket.connectToHost(url.host(),url.port(12345),QTcpSocket::ReadOnly);

    _start_recording = boost::posix_time::microsec_clock::local_time();
}


void NetworkRecorder::
        stopRecording()
{
    // TODO implement
    tcpSocket.disconnectFromHost();
}


bool NetworkRecorder::
        isStopped()
{
    // TODO implement
    return !tcpSocket.isOpen();
}


bool NetworkRecorder::
        canRecord()
{
    // TODO implement
    return true;
}


std::string NetworkRecorder::
        name()
{
    return "Network recording " + url.toString().toStdString();
}


float NetworkRecorder::
        sample_rate()
{
    return samplerate;
}


void NetworkRecorder::
        receivedData(const void*data, int byteCount)
{
    const short* shortdata = (const short*)data;
    int sampleCount = byteCount/sizeof(short);
    if (sampleCount*sizeof(short) != byteCount)
    {
        // TODO save these bytes and append them to the next burst
    }

    if (0 == sampleCount)
        return;

    long unsigned offset = actual_number_of_samples();

    // convert shortdata to normalized floats
    Signal::pBuffer b( new Signal::Buffer(offset, sampleCount, sample_rate() ) );
    float* p = b->waveform_data()->getCpuMemory();
    for (int i=0; i<sampleCount; ++i)
        p[i] = shortdata[i]/(float)SHRT_MAX;

    // add data
    QMutexLocker lock(&_data_lock);
    _last_update = boost::posix_time::microsec_clock::local_time();
     _data.put( b );
     lock.unlock();

     // notify listeners that we've got new data
    _postsink.invalidate_samples( Signal::Interval( offset, offset + sampleCount ));
}


void NetworkRecorder::
        readData()
{
    QByteArray byteArray;
    do
    {
        byteArray = tcpSocket.read(samplerate);
        receivedData(byteArray.data_ptr(), byteArray.size());
    } while (!byteArray.isEmpty());
}

} // namespace Adapters
