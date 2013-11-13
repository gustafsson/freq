#include "networkrecorder.h"

#include <QMutexLocker>
#include <QTcpSocket>
#include <QErrorMessage>

#include "TaskTimer.h"

namespace Adapters {

NetworkRecorder::NetworkRecorder(QUrl url, float samplerate)
:
    url(url),
    samplerate(samplerate)
{
    if (url.scheme() != "s16bursts")
    {
        // only our own special newly invented scheme 's16bursts' is supported
        QErrorMessage::qtHandler()->showMessage(
            QString("'%1' is not supported. Only the s16bursts:// protocol is supported.<br/><br/>"
            "Given url was: %2")
            .arg(url.scheme()).arg(url.toString()), "Network error");
        url = QUrl();
    }

    connect(&tcpSocket, SIGNAL(readyRead()), this, SLOT(readyRead()));
    connect(&tcpSocket, SIGNAL(connected()), this, SLOT(connected()));
    connect(&tcpSocket, SIGNAL(disconnected()), this, SLOT(disconnected()));
    connect(&tcpSocket, SIGNAL(hostFound()), this, SLOT(hostFound()));
    connect(&tcpSocket, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(error(QAbstractSocket::SocketError)));
    connect(&tcpSocket, SIGNAL(stateChanged(QAbstractSocket::SocketState)), this, SLOT(stateChanged(QAbstractSocket::SocketState)));
}


NetworkRecorder::~NetworkRecorder()
{
}


void NetworkRecorder::
        startRecording()
{
    _offset = actual_number_of_samples()/sample_rate();
    _start_recording = boost::posix_time::microsec_clock::local_time();

    tcpSocket.connectToHost(url.host(),url.port(12345),QTcpSocket::ReadOnly);
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
    switch(tcpSocket.state())
    {
    case QAbstractSocket::UnconnectedState:
        return true;
    default:
        return false;
    }
}


bool NetworkRecorder::
        canRecord()
{
    return !url.isEmpty();
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


float NetworkRecorder::
        length()
{
    return std::min( Signal::FinalSource::length(), time() );
}


float NetworkRecorder::
        time()
{
    switch(tcpSocket.state())
    {
    case QAbstractSocket::ConnectedState:
        return Recorder::time();
    default:
        return actual_number_of_samples()/sample_rate();
    }
}


int NetworkRecorder::
        receivedData(const void*data, int byteCount)
{
    const short* shortdata = (const short*)data;
    int sampleCount = byteCount/sizeof(short);

    if (0 == sampleCount)
        return 0;

    Signal::IntervalType offset = actual_number_of_samples();

    // convert shortdata to normalized floats
    Signal::pBuffer b( new Signal::Buffer(offset, sampleCount, sample_rate(), 1 ) );
    float* p = b->getChannel (0)->waveform_data()->getCpuMemory();
    for (int i=0; i<sampleCount; ++i)
        p[i] = shortdata[i]/(float)SHRT_MAX;

    // add data
    QMutexLocker lock(&_data_lock);
    _last_update = boost::posix_time::microsec_clock::local_time();
    _data.put( b );
    lock.unlock();

    // notify listeners that we've got new data
    _postsink.invalidate_samples( Signal::Interval( offset, offset + sampleCount ));

    return sampleCount*sizeof(short);
}


void NetworkRecorder::
        readyRead()
{
    TaskInfo("%s: %s", __FUNCTION__, url.toString().toStdString().c_str() );

    if (time()>length())
    {
        _offset = actual_number_of_samples()/sample_rate();
        _start_recording = boost::posix_time::microsec_clock::local_time();
        TaskInfo("%g > %g. Resetting clock from %g s", time(), length(), _offset);
    }

    QByteArray byteArray;
    do
    {
        byteArray = tcpSocket.read(samplerate);
        int readData = receivedData(byteArray.data_ptr(), byteArray.size());

        for (int i=byteArray.size()-1; i>=readData; --i)
            tcpSocket.ungetChar(byteArray[i]);
    } while (!byteArray.isEmpty());
}


void NetworkRecorder::
        connected()
{
    TaskInfo("%s: %s", __FUNCTION__, url.toString().toStdString().c_str() );
}


void NetworkRecorder::
        disconnected()
{
    TaskInfo("%s: %s", __FUNCTION__, url.toString().toStdString().c_str() );
}


void NetworkRecorder::
        hostFound()
{
    TaskInfo("%s: %s", __FUNCTION__, url.toString().toStdString().c_str() );
}


void NetworkRecorder::
        error(QAbstractSocket::SocketError error)
{
    TaskInfo("%s: %s - %s", __FUNCTION__,
             url.toString().toStdString().c_str(),
             tcpSocket.errorString().toStdString().c_str());

    switch(error)
    {
    case QAbstractSocket::SocketTimeoutError:
    case QAbstractSocket::NetworkError:
        QErrorMessage::qtHandler()->showMessage(
            QString("No response from %1<br/><br/>"
            "%2")
            .arg(url.toString())
            .arg(tcpSocket.errorString()), "Network error");
        break;

    default:
        QErrorMessage::qtHandler()->showMessage(
            QString("An error occured while recording from the network resource:<br/><br/>"
            "%1<br/><br/>"
            "Error: %2").arg(url.toString()).arg(tcpSocket.errorString()), "Network error");
        break;
    }

    // rely on stateChanged to notify listeners that something happened
}


void NetworkRecorder::
        stateChanged(QAbstractSocket::SocketState state)
{
    std::string stateString;

    switch(state)
    {
    case QAbstractSocket::UnconnectedState: stateString = "UnconnectedState"; break;
    case QAbstractSocket::HostLookupState:  stateString = "HostLookupState"; break;
    case QAbstractSocket::ConnectingState:  stateString = "ConnectingState"; break;
    case QAbstractSocket::ConnectedState:   stateString = "ConnectedState"; break;
    case QAbstractSocket::BoundState:       stateString = "BoundState"; break;
    case QAbstractSocket::ListeningState:   stateString = "ListeningState"; break;
    case QAbstractSocket::ClosingState:     stateString = "ClosingState"; break;
    default: stateString = "invalid state"; break;
    }

    TaskInfo("%s: %s - %s", __FUNCTION__,
             url.toString().toStdString().c_str(),
             stateString.c_str());

    // notify listeners that something happened (most meaningful if state == UnconnectedState)
    _postsink.invalidate_samples( Signal::Interval() );
}

} // namespace Adapters
