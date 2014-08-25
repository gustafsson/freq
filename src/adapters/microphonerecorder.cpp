#include "microphonerecorder.h"
#include "playback.h"
#include "sawe/configuration.h"

#include "tasktimer.h"
#include "demangle.h"

#include <iostream>
#include <memory.h>

#include <QMutexLocker>

#include <boost/foreach.hpp>

#define TIME_MICROPHONERECORDER
//#define TIME_MICROPHONERECORDER if(0)

//#define TIME_MICROPHONERECORDER_WRITEBUFFER
#define TIME_MICROPHONERECORDER_WRITEBUFFER if(0)

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

        TIME_MICROPHONERECORDER TaskTimer tt("Creating MicrophoneRecorder for device %d", input_device_);
        portaudio::System &sys = portaudio::System::instance();

        PaError err = Pa_Initialize();
        if (err != paNoError) {
            TaskInfo("MicrophoneRecorder: Pa_Initialize returned err = %d: %s", err, Pa_GetErrorText(err));
        }

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
        auto sample_rate = device.defaultSampleRate();
        unsigned num_channels = device.maxInputChannels();

        if (num_channels > 2)
            num_channels = 2;

        if (Sawe::Configuration::mono()) {
            if (num_channels > 1)
                num_channels = 1;
        }

        if (!_data || this->sample_rate() != sample_rate || this->num_channels() != num_channels)
            _data.reset (new Recorder::Data(sample_rate, num_channels));

        TIME_MICROPHONERECORDER TaskInfo(boost::format("Opening recording input stream on '%s' with %d"
                       " channels, %g samples/second"
                       " and input latency %g s or %g s")
                                         % device.name()
                                         % num_channels
                                         % sample_rate
                                         % device.defaultHighInputLatency()
                                         % device.defaultLowInputLatency ());

        _rolling_mean.resize(num_channels);
        for (unsigned i=0; i<num_channels; ++i)
            _rolling_mean[i] = 0;

        for (int interleaved=0; interleaved<2; ++interleaved)
        {
            _is_interleaved = interleaved!=0;

            portaudio::DirectionSpecificStreamParameters inParamsRecord(
                    device,
                    num_channels, // channels
                    portaudio::FLOAT32,
                    interleaved, // interleaved
        //#ifdef __APPLE__ // TODO document why
                    device.defaultHighInputLatency(),
        //#else
        //            device.defaultLowInputLatency(),
        //#endif
                    NULL);

            PaError err = Pa_IsFormatSupported(inParamsRecord.paStreamParameters(), 0, sample_rate);
            bool fmtok = err==paFormatIsSupported;
            if (!fmtok) {
                TaskInfo("MicrophoneRecorder: interleaved = %d not supported. err = %d: %s", interleaved, err, Pa_GetErrorText(err));
                continue;
            }

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

        if (!_stream_record) {
            TaskInfo("MicrophoneRecorder: Couldn't open recording stream");
            _has_input_device = false;
        }
    }
    catch (const portaudio::PaException& x)
    {
        TaskInfo("MicrophoneRecorder: PaException init error: %s %s (%d)\nMessage: %s",
                 vartype(x).c_str(), x.paErrorText(), x.paError(), x.what());
        _has_input_device = false;
    }
    catch (const portaudio::PaCppException& x)
    {
        TaskInfo("MicrophoneRecorder: PaCppException init error: %s (%d)\nMessage: %s",
                 vartype(x).c_str(), x.specifier(), x.what());
        _has_input_device = false;
    }
}


MicrophoneRecorder::~MicrophoneRecorder()
{
    stopRecording();

    auto d = _data.write ();
    auto& samples = d->samples;
    if (0<samples.spannedInterval ().count ()) {
        TIME_MICROPHONERECORDER TaskTimer tt("Releasing %s recorded data in %u channels",
                     Signal::SourceBase::lengthLongFormat ( samples.spannedInterval ().count ()/samples.sample_rate ()).c_str(),
                     samples.num_channels());
        samples.clear();
    }
}

void MicrophoneRecorder::startRecording()
{
    if (!stopping.valid ())
        return;

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

    _start_recording.restart ();
}

void MicrophoneRecorder::stopRecording()
{
    TIME_MICROPHONERECORDER TaskInfo ti("MicrophoneRecorder::stopRecording()");
    if (_stream_record)
    {
        decltype(_stream_record) sr;
        sr.swap (_stream_record);

        stopping = std::async (std::launch::async,
            [](decltype(_stream_record) sr, std::string deviceName)
            {
                try
                {
                TIME_MICROPHONERECORDER TaskInfo ti("Trying to stop recording on %s", deviceName.c_str());

                //stop could hang the ui (codaset #24)
                //_stream_record->isStopped()? void(): _stream_record->stop();

                // using abort instead of stop means that the recording thread will continue for a
                // while after abort has returned.
                sr->isStopped()? void(): sr->abort();

                sr->close();
                sr.reset();
                }
                catch (const portaudio::PaException& x)
                {
                    TaskInfo("stopRecording error: %s %s (%d)\nMessage: %s",
                             vartype(x).c_str(), x.paErrorText(), x.paError(), x.what());
                }
                catch (const portaudio::PaCppException& x)
                {
                    TaskInfo("stopRecording error: %s (%d)\nMessage: %s",
                             vartype(x).c_str(), x.specifier(), x.what());
                }
            }, std::move(sr), deviceName());
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


int MicrophoneRecorder::
        writeBuffer(const void *inputBuffer,
                 void * /*outputBuffer*/,
                 unsigned long framesPerBuffer,
                 const PaStreamCallbackTimeInfo * /*timeInfo*/,
                 PaStreamCallbackFlags /*statusFlags*/)
{
    try {
    Signal::IntervalType offset = actual_number_of_samples();

    float fs = _data.raw ()->sample_rate;
    unsigned nc = _data.raw ()->num_channels;

    _last_update.restart ();

    if (!_receive_buffer || _receive_buffer->number_of_samples ()!=(int)framesPerBuffer || _receive_buffer->number_of_channels ()!=nc)
        _receive_buffer = Signal::pBuffer( new Signal::Buffer(0, framesPerBuffer, 1, nc ) );

    _receive_buffer->set_sample_offset (offset);
    _receive_buffer->set_sample_rate (fs);

    Signal::Interval I = _receive_buffer->getInterval ();
    TIME_MICROPHONERECORDER_WRITEBUFFER TaskTimer tt(boost::format("MicrophoneRecorder: writeBuffer %s, [%g, %g) s")
                                        % I % (I.first/fs) % (I.last/fs));

    for (unsigned i=0; i<nc; ++i)
    {
        Signal::pMonoBuffer b = _receive_buffer->getChannel (i);
        float* p = CpuMemoryStorage::WriteAll<1>(b->waveform_data()).ptr ();
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

        // Not really a rolling mean, rather an IIR. It is anyway an approximated high-pass
        // filter at a few Hz, the microphone is not expected to such low frequencies
        float mean = _rolling_mean[in_i];
        for (unsigned j=0; j<framesPerBuffer; ++j)
        {
            float v = p[j];
            p[j] = v - mean;
            mean = mean*0.99999f + v*0.00001f;
        }
        _rolling_mean[in_i] = mean;
    }

    _data.write ()->samples.put( _receive_buffer );

    if (_invalidator)
        // Tell someone that there is new data available to read
        _invalidator.write ()->markNewlyRecordedData( _receive_buffer->getInterval () );

    } catch (...) {
        _exception = std::current_exception ();
        return paAbort;
    }

    return paContinue;
}

} // namespace Adapters



#include "timer.h"
#include "signal/recorderoperation.h"
#include <QSemaphore>

namespace Adapters {

class GotDataCallback: public Signal::Recorder::IGotDataCallback
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

void MicrophoneRecorder::
        test()
{
    // It should control the behaviour of a recording
    {
        int inputDevice = -1;
        Signal::Recorder::IGotDataCallback::ptr callback(new GotDataCallback);

        Signal::MicrophoneRecorderDesc mrd(Signal::Recorder::ptr(new MicrophoneRecorder(inputDevice)), callback);

        EXCEPTION_ASSERT( mrd.canRecord() );
        EXCEPTION_ASSERT( mrd.isStopped() );

        mrd.startRecording();

        EXCEPTION_ASSERT( !mrd.isStopped() );

        Timer t;
        dynamic_cast<GotDataCallback*>(callback.raw ())->wait (6000);
        // Re-throw exception if an exception was generated
        mrd.recorder()->read (Signal::Interval (0, 1));
        EXCEPTION_ASSERT_LESS( t.elapsed (), 1.200 );

        mrd.stopRecording();

        EXCEPTION_ASSERT( mrd.isStopped() );

        EXCEPTION_ASSERT(dynamic_cast<const GotDataCallback*>(&*callback.read ())->marked_data () != Signal::Intervals());
        EXCEPTION_ASSERT_LESS( t.elapsed (), 1.300 );
    }
}

} // namespace Signal
