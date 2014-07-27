#ifndef QTMICROPHONE_H
#define QTMICROPHONE_H

#include "signal/recorder.h"
#include <QSharedPointer>

class QAudioInput;
class QIODevice;

class QtMicrophone: public QObject, public Signal::Recorder
{
    Q_OBJECT
public:
    QtMicrophone();

    void startRecording() override;
    void stopRecording() override;
    bool isStopped() const override;
    bool canRecord() override;
    std::string name() override;

private slots:
    void gotData();

private:
    QSharedPointer<QAudioInput> audio_;
    QIODevice *device_ = 0;
    Signal::pBuffer buffer_;
    std::vector<float> samples_;
};

#endif // QTMICROPHONE_H
