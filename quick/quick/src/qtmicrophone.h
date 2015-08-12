#ifndef QTMICROPHONE_H
#define QTMICROPHONE_H

#include "signal/recorder.h"
#include <QScopedPointer>
#include <QObject>
#include <QAudioInput>
#include <QThread>

class QAudioInput;
class QIODevice;

class GotData : public QIODevice
{
    Q_OBJECT
public:
    GotData(shared_state<Signal::Recorder::Data> data,
            Signal::Processing::IInvalidator::ptr& invalidator,
            QAudioFormat format,
            QObject* parent=0);
    ~GotData();

    qint64 readData(char *data, qint64 maxlen);
    qint64 writeData(const char *data, qint64 len);

private:
    shared_state<Signal::Recorder::Data> data;
    Signal::Processing::IInvalidator::ptr& invalidator;
    QAudioFormat format;
    std::vector<float> f;
    Signal::pBuffer buffer;

    void writeData(const float* in, quint64 len);
};


class QtAudioObject: public QObject
{
    Q_OBJECT
public:
    QtAudioObject(QAudioDeviceInfo info, QAudioFormat format, QIODevice* device);
    ~QtAudioObject();

    bool isStopped();
    bool canRecord();

public slots:
    void init();
    void finished();
    void startRecording();
    void stopRecording();

private:
    QAudioDeviceInfo info_;
    QAudioFormat format_;
    QIODevice* device_ = 0;
    QAudioInput* audio_ = 0;
};


class QtMicrophone: public Signal::Recorder
{
public:
    QtMicrophone();
    ~QtMicrophone();

public:
    void startRecording() override;
    void stopRecording() override;
    bool isStopped() const override;
    bool canRecord() override;
    std::string name() override;

private:
    void init();

    QThread* audiothread_;
    QtAudioObject* audioobject_;

    void readSamples(unsigned n);
};

#endif // QTMICROPHONE_H
