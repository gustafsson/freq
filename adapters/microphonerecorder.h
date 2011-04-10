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
#include <boost/date_time/posix_time/posix_time.hpp>

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

    virtual std::string name();
    virtual Signal::pBuffer read( const Signal::Interval& I );
    virtual float sample_rate();
    virtual long unsigned number_of_samples();
    virtual unsigned num_channels();
    virtual void set_channel(unsigned c);
    virtual unsigned get_channel();

    unsigned recording_itr() { return number_of_samples(); }
    float time();
    float time_since_last_update();

    Signal::PostSink* getPostSink() { return &_postsink; }

private:
    MicrophoneRecorder()
        :
        input_device_(-1),
        _offset(0),
        _channel(0),
        _sample_rate(1)
    {} // for deserialization

    int input_device_;
    void init();

    boost::posix_time::ptime _start_recording, _last_update;
    float _offset;
    unsigned _channel;
    float _sample_rate;
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
        if (typename archive::is_saving())
            save_recording(ar, version);
        else
            load_recording(ar, version);
    }

    template<class archive>
    void save_recording(archive& ar, const unsigned int /*version*/)
    {
        // Save a microphonerecording as if it where an audiofile, save single channeled for now
        Signal::IntervalType N = number_of_samples();
        if (0==N) // workaround for the special case of saving an empty recording.
            N = 1;

        Signal::pBuffer b = readFixedLengthAllChannels( Signal::Interval(0, N) );
        std::stringstream ss;
        ss << "recording_" << std::oct << (void*)this << ".wav";

        WriteWav::writeToDisk(ss.str(), b, false);

        boost::shared_ptr<Audiofile> wavfile( new Audiofile(ss.str()) );

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);
        ar & BOOST_SERIALIZATION_NVP(wavfile);
        ar & BOOST_SERIALIZATION_NVP(input_device_);
    }

    template<class archive>
    void load_recording(archive& ar, const unsigned int /*version*/)
    {
        boost::shared_ptr<Audiofile> wavfile;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);
        ar & BOOST_SERIALIZATION_NVP(wavfile);
        ar & BOOST_SERIALIZATION_NVP(input_device_);

        init();

        Signal::Interval I(0, wavfile->number_of_samples());
        for (unsigned c=0; c<num_channels(); ++c)
        {
            wavfile->set_channel( c % wavfile->num_channels() );
            _data[c].put(wavfile->read( I ));
        }

        //init(-1);
    }
};

} // namespace Adapters

#endif // ADAPETERS_MICROPHONERECORDER_H
