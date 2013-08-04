#ifndef ADAPETERS_MICROPHONERECORDER_H
#define ADAPETERS_MICROPHONERECORDER_H

#include "writewav.h"
#include "audiofile.h"
#include "volatileptr.h"

#include "adapters/recorder.h"

#include <vector>
#include <sstream>
#include <stdio.h>

#include <portaudiocpp/PortAudioCpp.hxx>

#include <boost/scoped_ptr.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace Adapters {

class MicrophoneRecorder: public Recorder
{
public:
    MicrophoneRecorder(int inputDevice/*=-1*/);
    ~MicrophoneRecorder();

    virtual void startRecording();
    virtual void stopRecording();
    virtual bool isStopped();
    virtual bool canRecord();

    void changeInputDevice( int inputDevice );
    void setProjectName(std::string, int);

    virtual std::string name();
    virtual float sample_rate();

private:
    MicrophoneRecorder()
        :
        input_device_(-1),
        _sample_rate(1),
        _is_interleaved(false)
    {} // for deserialization

    std::string deviceName();
    void init();

    int input_device_;
    float _sample_rate;
    bool _is_interleaved;
    bool _has_input_device;
    std::vector<float> _rolling_mean;

    // todo remove Sink* _callback;
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

        Signal::pBuffer b = readFixedLength( Signal::Interval(0, N) );

        WriteWav::writeToDisk(_filename, b, false);

        boost::shared_ptr<Audiofile> wavfile( new Audiofile(_filename) );

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(DeprecatedOperation);
        ar & BOOST_SERIALIZATION_NVP(wavfile);
        ar & BOOST_SERIALIZATION_NVP(input_device_);

        ::remove( _filename.c_str() );
    }

    template<class archive>
    void load_recording(archive& ar, const unsigned int /*version*/)
    {
        boost::shared_ptr<Audiofile> wavfile;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(DeprecatedOperation);
        ar & BOOST_SERIALIZATION_NVP(wavfile);
        ar & BOOST_SERIALIZATION_NVP(input_device_);

        init();

        _data.put(wavfile->readFixedLength( wavfile->getInterval() ));
    }
};


/**
 * @brief The MicrophoneRecorderOperation class should provide access to recorded data.
 */
class MicrophoneRecorderOperation: public Signal::Operation
{
public:
    MicrophoneRecorderOperation( Signal::pOperation recorder_ );

    virtual Signal::pBuffer process(Signal::pBuffer b);

private:
    Signal::pOperation recorder_;
};


/**
 * @brief The MicrophoneRecorderDesc class should control the behaviour of a recording.
 */
class MicrophoneRecorderDesc: public Signal::OperationDesc
{
public:
    class IGotDataCallback: public VolatilePtr<IGotDataCallback>
    {
    public:
        virtual void markNewlyRecordedData(Signal::Interval what)=0;
    };

    MicrophoneRecorderDesc( int inputDevice, IGotDataCallback::Ptr invalidator );

    void changeInputDevice( int inputDevice );
    void startRecording();
    void stopRecording();
    bool isStopped();
    bool canRecord();
    MicrophoneRecorder* recorder() const;

    // OperationDesc
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    virtual Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    virtual OperationDesc::Ptr copy() const;
    virtual Signal::Operation::Ptr createOperation( Signal::ComputingEngine* engine ) const;
    virtual Extent extent() const;

private:
    void setDataCallback( IGotDataCallback::Ptr invalidator );

    Signal::pOperation recorder_;
    int input_device_;
    IGotDataCallback::Ptr invalidator_;

public:
    static void test();
};

} // namespace Adapters

#endif // ADAPETERS_MICROPHONERECORDER_H
