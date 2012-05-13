#ifndef NETWORKRECORDER_H
#define NETWORKRECORDER_H

#include "recorder.h"

#include <QUrl>
#include <QTcpSocket>
#include <QHostAddress>

namespace Adapters {

/**
  To use 'NetworkRecorder' start Sonic AWE with the syntax below and hit the regular 'record' button.

  sonicawe s16bursts://hostname:hostport

  Example:

  sonicawe s16bursts://123.45.67.89:12345


  NetworkRecorder only supports data in signed 16-bit integers.
  */
class NetworkRecorder: public QObject, public Recorder
{
    Q_OBJECT
public:
    NetworkRecorder(QUrl url, float samplerate=32768 );
    ~NetworkRecorder();

    virtual void startRecording();
    virtual void stopRecording();
    virtual bool isStopped();
    virtual bool canRecord();

    virtual std::string name();
    virtual float sample_rate();
    virtual float length();

private:
    QUrl url;
    QTcpSocket tcpSocket;
    float samplerate;

    virtual float time();
    int receivedData(const void*data, int byteCount);

private slots:
    void readyRead();
    void connected();
    void disconnected();
    void hostFound();
    void error(QAbstractSocket::SocketError);
    void stateChanged(QAbstractSocket::SocketState);
};

} // namespace Adapters

#endif // NETWORKRECORDER_H
