#include "microphonerecorder.h"
#include "playback.h"
#include "sawe/configuration.h"

#include "tasktimer.h"
#include "demangle.h"

#include <iostream>
#include <memory.h>

#include <QMutexLocker>

#include <boost/foreach.hpp>

//#define TIME_MICROPHONERECORDER
#define TIME_MICROPHONERECORDER if(0)

using namespace std;

namespace Adapters {


MicrophoneRecorder::
        MicrophoneRecorder(int inputDevice)
            :
            input_device_(inputDevice),
            _is_interleaved(false)
{
    std::stringstream ss;
    ::srand(::time(0));
    unsigned int s = rand()^(rand() << 8) ^ (rand() << 16) ^ (rand() << 24);

    ss << "recording_" << s << ".wav";

    init(); // fetch _sample_rate and _num_channels
    stopRecording(); // delete _stream_record
}


void MicrophoneRecorder::
        init()
{
    try
    {
        _offset = 0;
        _sample_rate = 1;
        _num_channels = 0;

        TIME_MICROPHONERECORDER TaskTimer tt("Creating MicrophoneRecorder for device %d", input_device_);
        portaudio::System &sys = portaudio::System::instance();

        _has_input_device = false;
        for (int i=0; i < sys.deviceCount(); ++i)
        {
            if (!sys.deviceByIndex(i).isOutputOnlyDevice())
                _has_input_device = true;
        }

        if (!_has_input_device)
        {
            TaskInfo("System didn't report any recording devices. Can't record.");
            return;
        }

        if (0>input_device_) {
            input_device_ = sys.defaultInputDevice().index();
        } else if (input_device_ >= sys.deviceCount()) {
            input_device_ = sys.defaultInputDevice().index();
            TaskInfo("Total number of devices is %d, reverting to default input device %d", sys.deviceCount(), input_device_);
        } else if ( sys.deviceByIndex(input_device_).isOutputOnlyDevice() ) {
            TaskInfo(boost::format("Requested device '%s' can only be used for output") % sys.deviceByIndex(input_device_).name());
            input_device_ = sys.defaultInputDevice().index();
        } else {
            ;
        }

        TIME_MICROPHONERECORDER TaskInfo(boost::format("Using device '%s' for audio input") % sys.deviceByIndex(input_device_).name());

        portaudio::Device& device = sys.deviceByIndex(input_device_);
        _sample_rate = device.defaultSampleRate();

        _num_channels = device.maxInputChannels();
        if (_num_channels > 2)
            _num_channels = 2;

        if (Sawe::Configuration::mono()) {
            if (_num_channels > 1)
                _num_channels = 1;
        }

        TIME_MICROPHONERECORDER TaskInfo(boost::format("Opening recording input stream on '%s' with %d"
                       " channels, %g samples/second"
                       " and input latency %g s")
                                         % device.name()
                                         % _num_channels
                                         % _sample_rate
                                         % device.defaultHighInputLatency());

        _rolling_mean.resize(_num_channels);
        for (unsigned i=0; i<_num_channels; ++i)
            _rolling_mean[i] = 0;

        for (int interleaved=0; interleaved<2; ++interleaved)
        {
            _is_interleaved = interleaved!=0;

            portaudio::DirectionSpecificStreamParameters inParamsRecord(
                    device,
                    _num_channels, // channels
                    portaudio::FLOAT32,
                    interleaved, // interleaved
        //#ifdef __APPLE__ // TODO document why
                    device.defaultHighInputLatency(),
        //#else
        //            device.defaultLowInputLatency(),
        //#endif
                    NULL);

            PaError err = Pa_IsFormatSupported(inParamsRecord.paStreamParameters(), 0, device.defaultSampleRate());
            bool fmtok = err==paFormatIsSupported;
            if (!fmtok)
                continue;

            portaudio::StreamParameters paramsRecord(
                    inParamsRecord,
                    portaudio::DirectionSpecificStreamParameters::null(),
                    device.defaultSampleRate(),
                    paFramesPerBufferUnspecified,
                    paNoFlag);

            _stream_record.reset( new portaudio::MemFunCallbackStream<MicrophoneRecorder>(
                    paramsRecord,
                    *this,
                    &MicrophoneRecorder::writeBuffer));
            break;
        }

        if (!_stream_record)
            _has_input_device = false;
    }
    catch (const portaudio::PaException& x)
    {
        TaskInfo("a2 MicrophoneRecorder init error: %s %s (%d)\nMessage: %s",
                 vartype(x).c_str(), x.paErrorText(), x.paError(), x.what());
        _has_input_device = false;
    }
    catch (const portaudio::PaCppException& x)
    {
        TaskInfo("b2 MicrophoneRecorder init error: %s (%d)\nMessage: %s",
                 vartype(x).c_str(), x.specifier(), x.what());
        _has_input_device = false;
    }
}


MicrophoneRecorder::~MicrophoneRecorder()
{
    stopRecording();

    QMutexLocker lock(&_data_lock);
    if (0<_data.spannedInterval ().count ()) {
        TIME_MICROPHONERECORDER TaskTimer tt("Releasing %s recorded data in %u channels",
                     Signal::SourceBase::lengthLongFormat ( _data.spannedInterval ().count ()/_data.sample_rate ()).c_str(),
                     _data.num_channels());
        _data.clear();
    }
}

void MicrophoneRecorder::startRecording()
{
    TIME_MICROPHONERECORDER TaskInfo ti("MicrophoneRecorder::startRecording()");
    init();

    if (!canRecord())
    {
        TaskInfo("MicrophoneRecorder::startRecording() cant start recording");
        return;
    }

    // call length() and update _offset before starting recording because
    // length() uses {number_of_samples - time() - _offset} while recording.
    _offset = length();

    try
    {
        _stream_record->start();
    }
    catch (const portaudio::PaException& x)
    {
        TaskInfo("startRecording error: %s %s (%d)\nMessage: %s",
                 vartype(x).c_str(), x.paErrorText(), x.paError(), x.what());
        _has_input_device = false;
        return;
    }
    catch (const portaudio::PaCppException& x)
    {
        TaskInfo("startRecording error: %s (%d)\nMessage: %s",
                 vartype(x).c_str(), x.specifier(), x.what());
        _has_input_device = false;
        return;
    }

    _start_recording = boost::posix_time::microsec_clock::local_time();
}

void MicrophoneRecorder::stopRecording()
{
    TIME_MICROPHONERECORDER TaskInfo ti("MicrophoneRecorder::stopRecording()");
    if (_stream_record) {
        try
        {
        TIME_MICROPHONERECORDER TaskInfo ti("Trying to stop recording on %s", deviceName().c_str());
        //stop could hang the ui (codaset #24)
        //_stream_record->isStopped()? void(): _stream_record->stop();
        _stream_record->isStopped()? void(): _stream_record->abort();
        _stream_record->close();
        _stream_record.reset();
        }
        catch (const portaudio::PaException& x)
        {
            TaskInfo("stopRecording error: %s %s (%d)\nMessage: %s",
                     vartype(x).c_str(), x.paErrorText(), x.paError(), x.what());
            _has_input_device = false;
        }
        catch (const portaudio::PaCppException& x)
        {
            TaskInfo("stopRecording error: %s (%d)\nMessage: %s",
                     vartype(x).c_str(), x.specifier(), x.what());
            _has_input_device = false;
        }
    }
}

bool MicrophoneRecorder::isStopped() const
{
    return _stream_record?_stream_record->isStopped():true;
}

std::string MicrophoneRecorder::
        deviceName()
{
    portaudio::System &sys = portaudio::System::instance();
    int d = input_device_;
    if (d<0 || d>=sys.deviceCount())
        d = sys.defaultInputDevice().index();

    return sys.deviceByIndex(d).name();
}

void MicrophoneRecorder::
        setProjectName(std::string project, int i)
{
    stringstream ss;
    string device = deviceName();
    replace(device.begin(), device.end(), ' ', '_');
    ss << project << "_" << i << "_" << device << ".wav";
    _filename = ss.str();
}


bool MicrophoneRecorder::
        canRecord()
{
    return _has_input_device;
}


void MicrophoneRecorder::
        changeInputDevice( int inputDevice )
{
    bool isRecording = !isStopped();
    stopRecording();

    input_device_ = inputDevice;
    init();

    if (isRecording)
        startRecording();
}


std::string MicrophoneRecorder::
        name()
{
    return "Recording mic " + deviceName();
}


float MicrophoneRecorder::
        sample_rate() const
{
    return _sample_rate;
}


unsigned MicrophoneRecorder::
        num_channels() const
{
    return _num_channels;
}


int MicrophoneRecorder::
        writeBuffer(const void *inputBuffer,
                 void * /*outputBuffer*/,
                 unsigned long framesPerBuffer,
                 const PaStreamCallbackTimeInfo * /*timeInfo*/,
                 PaStreamCallbackFlags /*statusFlags*/)
{
    TIME_MICROPHONERECORDER TaskTimer tt("MicrophoneRecorder::writeBuffer(%u new samples) inputBuffer = %p", framesPerBuffer, inputBuffer);

    Signal::IntervalType offset = actual_number_of_samples();
    QMutexLocker lock(&_data_lock);
    _last_update = boost::posix_time::microsec_clock::local_time();

    Signal::pBuffer mb( new Signal::Buffer(0, framesPerBuffer, sample_rate(), _num_channels ) );
    for (unsigned i=0; i<_num_channels; ++i)
    {
        Signal::pMonoBuffer b = mb->getChannel (i);
        float* p = b->waveform_data()->getCpuMemory();
        unsigned in_num_channels = _rolling_mean.size ();
        unsigned in_i = i;
        if (in_i >= in_num_channels)
            in_i = in_num_channels - 1;

        if (_is_interleaved)
        {
            const float *in = (const float *)inputBuffer;
            for (unsigned j=0; j<framesPerBuffer; ++j)
                p[j] = in[j*in_num_channels + i];
        }
        else
        {
            const float **in = (const float **)inputBuffer;
            const float *buffer = in[in_i];
            for (unsigned j=0; j<framesPerBuffer; ++j)
                p[j] = buffer[j];
        }

        float& mean = _rolling_mean[in_i];
        for (unsigned j=0; j<framesPerBuffer; ++j)
        {
            float v = p[j];
            p[j] = v - mean;
            mean = mean*0.99999f + v*0.00001f;
        }

//        memcpy ( b->waveform_data()->getCpuMemory(),
//                 buffer,
//                 framesPerBuffer*sizeof(float) );

        b->set_sample_offset( offset );
        b->set_sample_rate( sample_rate() );

        TIME_MICROPHONERECORDER TaskInfo ti("Interval: %s, [%g, %g) s",
                                            b->getInterval().toString().c_str(),
                                            b->getInterval().first / b->sample_rate(),
                                            b->getInterval().last / b->sample_rate() );

    }

    _data.put( mb );
    lock.unlock();

    if (_invalidator)
        _invalidator.write ()->markNewlyRecordedData( Signal::Interval( offset, offset + framesPerBuffer ) );

    return paContinue;
}


MicrophoneRecorderOperation::
        MicrophoneRecorderOperation( Recorder::Ptr recorder )
    :
      recorder_(recorder)
{
}


Signal::pBuffer MicrophoneRecorderOperation::
        process(Signal::pBuffer b)
{
    return recorder_.write ()->read (b->getInterval ());
}


MicrophoneRecorderDesc::
        MicrophoneRecorderDesc(Recorder::Ptr recorder, Recorder::IGotDataCallback::Ptr invalidator)
    :
      recorder_(recorder)
{
    recorder_.write ()->setDataCallback(invalidator);
}


void MicrophoneRecorderDesc::
        startRecording()
{
    recorder_.write ()->startRecording ();
}


void MicrophoneRecorderDesc::
        stopRecording()
{
    recorder_.write ()->stopRecording ();
}


bool MicrophoneRecorderDesc::
        isStopped()
{
    return recorder_.write ()->isStopped ();
}


bool MicrophoneRecorderDesc::
        canRecord()
{
    return recorder_.write ()->canRecord ();
}


Signal::Interval MicrophoneRecorderDesc::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Signal::Interval MicrophoneRecorderDesc::
        affectedInterval( const Signal::Interval& I ) const
{
    return I;
}


Signal::OperationDesc::Ptr MicrophoneRecorderDesc::
        copy() const
{
    EXCEPTION_ASSERTX(false, "Can't make a copy of microphone recording");
    return Signal::OperationDesc::Ptr();
}


Signal::Operation::Ptr MicrophoneRecorderDesc::
        createOperation(Signal::ComputingEngine*) const
{
    Signal::Operation::Ptr r(new MicrophoneRecorderOperation(recorder_));
    return r;
}


MicrophoneRecorderDesc::Extent MicrophoneRecorderDesc::
        extent() const
{
    auto rec = recorder_.read ();
    MicrophoneRecorderDesc::Extent x;
    x.interval = Signal::Interval(0, rec->number_of_samples());
    x.number_of_channels = rec->num_channels ();
    x.sample_rate = rec->sample_rate ();
    return x;
}


Recorder::Ptr MicrophoneRecorderDesc::
        recorder() const
{
    return recorder_;
}


} // namespace Adapters


#include "timer.h"

#include <QSemaphore>


namespace Adapters {

class GotDataCallback: public Recorder::IGotDataCallback
{
public:
    Signal::Intervals marked_data() const { return marked_data_; }

    virtual void markNewlyRecordedData(Signal::Interval what) {
        marked_data_ |= what;
        semaphore_.release ();
    }

    void wait(int ms_timeout) {
        semaphore_.tryAcquire (1, ms_timeout);
    }

private:
    QSemaphore semaphore_;
    Signal::Intervals marked_data_;
};

void MicrophoneRecorderDesc::
        test()
{
    // It should control the behaviour of a recording
    {
        int inputDevice = -1;
        Recorder::IGotDataCallback::Ptr callback(new GotDataCallback);

        MicrophoneRecorderDesc mrd(Recorder::Ptr(new MicrophoneRecorder(inputDevice)), callback);

        EXCEPTION_ASSERT( mrd.canRecord() );
        EXCEPTION_ASSERT( mrd.isStopped() );

        mrd.startRecording();

        EXCEPTION_ASSERT( !mrd.isStopped() );

        Timer t;
        dynamic_cast<GotDataCallback*>(callback.raw ())->wait (6000);
        EXCEPTION_ASSERT_LESS( t.elapsed (), 1.200 );

        mrd.stopRecording();

        EXCEPTION_ASSERT( mrd.isStopped() );

        EXCEPTION_ASSERT(dynamic_cast<const GotDataCallback*>(&*callback.read ())->marked_data () != Signal::Intervals());
        EXCEPTION_ASSERT_LESS( t.elapsed (), 1.300 );
    }
}

} // namespace Adapters
