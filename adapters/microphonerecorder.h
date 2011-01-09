#ifndef ADAPETERS_MICROPHONERECORDER_H
#define ADAPETERS_MICROPHONERECORDER_H

#include "writewav.h"
#include "audiofile.h"

#include "signal/sinksource.h"
#include "signal/postsink.h"

#include <vector>
#include <sstream>

#include <portaudiocpp/PortAudioCpp.hxx>

#include <boost/scoped_ptr.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <QMutex>

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
    virtual float sample_rate();
    virtual long unsigned number_of_samples();
    virtual unsigned num_channels();
    virtual void set_channel(unsigned c);
    virtual unsigned get_channel();

    unsigned recording_itr() { return number_of_samples(); }

    Signal::PostSink* getPostSink() { return &_postsink; }

private:
    MicrophoneRecorder() {} // for deserialization
    void init(int inputDevice);

    unsigned channel;
    QMutex _data_lock;
    std::vector<Signal::SinkSource> _data;
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

    template<class archive>
    void serialize(archive& ar, const unsigned int version)
    {
        //if (ar.is_saving())
        if (_data.size())
            save_recording(ar, version);
        else
            load_recording(ar, version);
    }

    template<class archive>
    void save_recording(archive& ar, const unsigned int /*version*/)
    {
        // Save a microphonerecording as if it where an audiofile, save single channeled for now
        Signal::pBuffer b = readFixedLength( Signal::Interval(0, number_of_samples()) );
        std::stringstream ss;
        ss << "recording_" << std::oct << (void*)this << ".wav";

        WriteWav::writeToDisk(ss.str(), b, false);

        boost::shared_ptr<Audiofile> wavfile( new Audiofile(ss.str()) );

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);
        ar & BOOST_SERIALIZATION_NVP(wavfile);
    }

    template<class archive>
    void load_recording(archive& ar, const unsigned int /*version*/)
    {
        boost::shared_ptr<Audiofile> wavfile;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);
        ar & BOOST_SERIALIZATION_NVP(wavfile);

        init(-1);

        Signal::Interval I(0, wavfile->number_of_samples());
        for (unsigned c=0; c<num_channels(); ++c)
        {
            _data[c].put(wavfile->read( I ));
        }

        //init(-1);
    }
};

} // namespace Adapters

#endif // ADAPETERS_MICROPHONERECORDER_H