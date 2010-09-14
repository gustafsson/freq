#ifndef SIGNALMICROPHONERECORDER_H
#define SIGNALMICROPHONERECORDER_H

#include <vector>
#include <QMutex>
#include "signal/sinksource.h"
#include <portaudiocpp/PortAudioCpp.hxx>
#include <QObject>

namespace Signal {

/**
QObject has to be first.
  */
class MicrophoneRecorder: public QObject, public FinalSource
{
    Q_OBJECT
public:
    MicrophoneRecorder(int inputDevice/*=-1*/);
    ~MicrophoneRecorder();

    void startRecording();
    void stopRecording();
    bool isStopped();

    virtual pBuffer read( const Interval& I );
    virtual unsigned sample_rate();
    virtual long unsigned number_of_samples();

    unsigned recording_itr() { return number_of_samples(); }

//signals:
//    void data_available(MicrophoneRecorder*);

private:
    SinkSource _data;

    // todo remove Sink* _callback;
    portaudio::AutoSystem _autoSys;
    boost::scoped_ptr<portaudio::MemFunCallbackStream<MicrophoneRecorder> > _stream_record;

    int writeBuffer(const void *inputBuffer,
                     void * /*outputBuffer*/,
                     unsigned long framesPerBuffer,
                     const PaStreamCallbackTimeInfo *timeInfo,
                     PaStreamCallbackFlags statusFlags);

    friend class boost::serialization::access;
    template<class archive> void save(archive& ar, const unsigned int version) {
        throw std::logic_error("don't know how to save a microphonerecording");
    }
    template<class archive> void load(archive& ar, const unsigned int version) {
        throw std::logic_error("don't know how to load microphonerecording");
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

} // namespace Waveform

#endif // SIGNALMICROPHONERECORDER_H
