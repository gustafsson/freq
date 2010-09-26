#ifndef ADAPETERS_MICROPHONERECORDER_H
#define ADAPETERS_MICROPHONERECORDER_H

#include "signal/sinksource.h"
#include "signal/postsink.h"

#include <vector>
#include <QMutex>
#include <portaudiocpp/PortAudioCpp.hxx>

namespace Adapters {

class MicrophoneRecorder: public Signal::FinalSource
{
public:
    MicrophoneRecorder(int inputDevice/*=-1*/);
    ~MicrophoneRecorder();

    void startRecording();
    void stopRecording();
    bool isStopped();

    virtual Signal::pBuffer read( const Signal::Interval& I );
    virtual unsigned sample_rate();
    virtual long unsigned number_of_samples();

    unsigned recording_itr() { return number_of_samples(); }

    Signal::PostSink* getPostSink() { return &_postsink; }

private:
    Signal::SinkSource _data;
    Signal::PostSink _postsink;

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

} // namespace Adapters

#endif // ADAPETERS_MICROPHONERECORDER_H
