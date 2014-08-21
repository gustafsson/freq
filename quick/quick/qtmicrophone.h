#ifndef QTMICROPHONE_H
#define QTMICROPHONE_H

#include "signal/recorder.h"
#include <QScopedPointer>
#include <QObject>
#include <QAudioInput>

class QAudioInput;
class QIODevice;

class GotData : public QIODevice
{
    Q_OBJECT
public:
    GotData(shared_state<Signal::Recorder::Data> data,
            Signal::Recorder::IGotDataCallback::ptr& invalidator,
            QAudioFormat format,
            QObject* parent=0);
    ~GotData();

    qint64 readData(char *data, qint64 maxlen);
    qint64 writeData(const char *data, qint64 len);

private:
    shared_state<Signal::Recorder::Data> data;
    Signal::Recorder::IGotDataCallback::ptr& invalidator;
    QAudioFormat format;
    std::vector<float> f;
    Signal::pBuffer buffer;

    void writeData(const float* in, quint64 len);
};


class QtMicrophone: public Signal::Recorder
{
public:
    QtMicrophone();
    ~QtMicrophone();

    void startRecording() override;
    void stopRecording() override;
    bool isStopped() const override;
    bool canRecord() override;
    std::string name() override;

private:
    QScopedPointer<QAudioInput> audio_;
    QScopedPointer<QIODevice> device_;

    void readSamples(unsigned n);
};

#endif // QTMICROPHONE_H
