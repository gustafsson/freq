#ifndef WAVEFORMRECORDER_H
#define WAVEFORMRECORDER_H

#include <vector>
#include "signal-source.h"
#include <portaudiocpp/PortAudioCpp.hxx>

namespace Signal {

class MicrophoneRecorder: public Source
{
public:
    class Callback {
        public:
        virtual void recievedData( MicrophoneRecorder* );
    };

    MicrophoneRecorder();
    ~MicrophoneRecorder();

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );
    virtual unsigned sample_rate();
    virtual unsigned number_of_samples();

    unsigned recording_itr() { return number_of_samples(); }

    void setCallback( Callback* );

private:
    Callback* _callback;
    portaudio::AutoSystem _autoSys;
    boost::scoped_ptr<portaudio::MemFunCallbackStream<MicrophoneRecorder> > _stream_record;

    std::vector<pBuffer> _cache;

    int writeBuffer(const void *inputBuffer,
                     void */*outputBuffer*/,
                     unsigned long framesPerBuffer,
                     const PaStreamCallbackTimeInfo *timeInfo,
                     PaStreamCallbackFlags statusFlags);

    unsigned _sample_rate;
};

} // namespace Waveform

#endif // WAVEFORMRECORDER_H
