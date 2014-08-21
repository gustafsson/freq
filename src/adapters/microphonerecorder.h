#ifndef ADAPETERS_MICROPHONERECORDER_H
#define ADAPETERS_MICROPHONERECORDER_H

#include "writewav.h"
#include "audiofile.h"
#include "shared_state.h"
#include "verifyexecutiontime.h"

#include "signal/recorder.h"

#include <vector>
#include <sstream>
#include <stdio.h>

#include <portaudiocpp/PortAudioCpp.hxx>

#include <boost/scoped_ptr.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace Adapters {

class MicrophoneRecorder: public Signal::Recorder
{
public:
    MicrophoneRecorder(int inputDevice/*=-1*/);
    ~MicrophoneRecorder();

    virtual void startRecording() override;
    virtual void stopRecording() override;
    virtual bool isStopped() const override;
    virtual bool canRecord() override;
    virtual std::string name() override;

    void changeInputDevice( int inputDevice );
    void setProjectName(std::string, int);

private:
    MicrophoneRecorder()
        :
        input_device_(-1),
        _is_interleaved(false)
    {} // for deserialization

    std::string deviceName();
    void init();

    int input_device_;
    bool _is_interleaved;
    bool _has_input_device;
    std::vector<float> _rolling_mean;
    Signal::pBuffer _receive_buffer;

    portaudio::AutoSystem _autoSys;
    boost::scoped_ptr<portaudio::MemFunCallbackStream<MicrophoneRecorder> > _stream_record;

    int writeBuffer(const void *inputBuffer,
                     void * /*outputBuffer*/,
                     unsigned long framesPerBuffer,
                     const PaStreamCallbackTimeInfo *timeInfo,
                     PaStreamCallbackFlags statusFlags);

    std::string _filename;

    friend class boost::serialization::access;

    template<class archive>
    void serialize(archive& ar, const unsigned int version)
    {
        if (typename archive::is_saving())
            save_recording(ar, version);
        else
            load_recording(ar, version);
    }

    template<class archive>
    void save_recording(archive& ar, const unsigned int /*version*/)
    {
        // Save a microphonerecording as if it were an audiofile, save single channeled for now
        Signal::IntervalType N = number_of_samples();
        if (0==N) // workaround for the special case of saving an empty recording.
            N = 1;

        Signal::pBuffer b = read( Signal::Interval(0, N) );

        WriteWav::writeToDisk(_filename, b, false);

        boost::shared_ptr<Audiofile> wavfile( new Audiofile(_filename) );

        ar & BOOST_SERIALIZATION_NVP(wavfile);
        ar & BOOST_SERIALIZATION_NVP(input_device_);

        ::remove( _filename.c_str() );
    }

    template<class archive>
    void load_recording(archive& ar, const unsigned int /*version*/)
    {
        boost::shared_ptr<Audiofile> wavfile;

        ar & BOOST_SERIALIZATION_NVP(wavfile);
        ar & BOOST_SERIALIZATION_NVP(input_device_);

        init();

        _data->samples.put(wavfile->readFixedLength( wavfile->getInterval() ));
    }

public:
    static void test();
};

} // namespace Adapters

#endif // ADAPETERS_MICROPHONERECORDER_H
