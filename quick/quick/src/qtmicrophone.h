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


class QtMicrophone: public QObject, public Signal::Recorder
{
    Q_OBJECT
public:
    QtMicrophone();
    ~QtMicrophone();

public slots:
    void startRecording() override;
    void stopRecording() override;

public:
    bool isStopped() const override;
    bool canRecord() override;
    std::string name() override;

private slots:
    void init();
    void finished();

private:
    QThread audiothread_;
    QAudioInput* audio_;
    QIODevice* device_;

    void readSamples(unsigned n);
};

#endif // QTMICROPHONE_H
