#include "microphonerecorder.h"
#include "playback.h"

#include <iostream>
#include <memory.h>

#include <QMutexLocker>

#include <Statistics.h>

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

    init(); // fetch _sample_rate
    stopRecording(); // delete _stream_record
}


void MicrophoneRecorder::
        init()
{
    try
    {
        // To avoid division with zero and compute the actual length as 0 when
        // dividing the number of samples with sample rate.
        _offset = 0;
        _sample_rate = 1;

        static bool first = true;
        if (first) Playback::list_devices();

        TaskTimer tt("Creating MicrophoneRecorder for device %d", input_device_);
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
            tt.getStream() << "Requested device '" << sys.deviceByIndex(input_device_).name() << "' can only be used for output";
            input_device_ = sys.defaultInputDevice().index();
        } else {
            ;
        }

        tt.getStream() << "Using device '" << sys.deviceByIndex(input_device_).name() << "' for audio input";

        portaudio::Device& device = sys.deviceByIndex(input_device_);
        _sample_rate = device.defaultSampleRate();

        unsigned channel_count = device.maxInputChannels();
        if (channel_count>2)
            channel_count = 2;
        tt.getStream() << "Opening recording input stream on '" << device.name() << "' with " << channel_count
                       << " channels, " << device.defaultSampleRate() << " samples/second"
                       << " and input latency " << device.defaultHighInputLatency() << " s";

        QMutexLocker lock(&_data_lock);
        if (_rolling_mean.empty ())
            _data = Signal::SinkSource(channel_count);

        _rolling_mean.resize(channel_count);
        for (unsigned i=0; i<channel_count; ++i)
            _rolling_mean[i] = 0;

        for (int interleaved=0; interleaved<2; ++interleaved)
        {
            _is_interleaved = interleaved!=0;

            portaudio::DirectionSpecificStreamParameters inParamsRecord(
                    device,
                    channel_count, // channels
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
	TaskTimer tt("%s", __FUNCTION__);

    stopRecording();

    QMutexLocker lock(&_data_lock);
    if (0<_data.length()) {
        TaskTimer tt("Releasing %s recorded data in %u channels",
                     _data.lengthLongFormat().c_str(),
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
        TaskInfo ti("Trying to stop recording on %s", deviceName().c_str());
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

bool MicrophoneRecorder::isStopped()
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
        sample_rate()
{
    return _sample_rate;
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
    unsigned num_channels = _data.num_channels();

    Signal::pBuffer mb( new Signal::Buffer(0, framesPerBuffer, sample_rate(), num_channels ) );
    for (unsigned i=0; i<num_channels; ++i)
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

    _postsink.invalidate_samples( Signal::Interval( offset, offset + framesPerBuffer ));

    return paContinue;
}


MicrophoneRecorderOperation::
        MicrophoneRecorderOperation( Signal::pOperation recorder )
    :
      recorder_(recorder)
{
}


Signal::pBuffer MicrophoneRecorderOperation::
        process(Signal::pBuffer b)
{
    return recorder_->readFixedLength (b->getInterval ());
}


class MarshallNewlyRecordedData: public Signal::Sink {
public:
    MarshallNewlyRecordedData(MicrophoneRecorderDesc::IGotDataCallback::Ptr invalidator)
        :
          invalidator_(invalidator)
    {}

    virtual void invalidate_samples(const Signal::Intervals& I) {
        BOOST_FOREACH(const Signal::Interval& i, I)
            write1(invalidator_)->markNewlyRecordedData(i);
    }
    virtual Signal::Intervals invalid_samples() {return Signal::Intervals(); }

private:
    MicrophoneRecorderDesc::IGotDataCallback::Ptr invalidator_;
};


MicrophoneRecorderDesc::
        MicrophoneRecorderDesc(int inputDevice, IGotDataCallback::Ptr invalidator)
    :
      recorder_(new MicrophoneRecorder(inputDevice)),
      input_device_(inputDevice),
      invalidator_(invalidator)
{
    setDataCallback(invalidator);
}


void MicrophoneRecorderDesc::
        changeInputDevice( int inputDevice )
{
    recorder()->changeInputDevice (inputDevice);
}


void MicrophoneRecorderDesc::
        startRecording()
{
    recorder()->startRecording ();
}


void MicrophoneRecorderDesc::
        stopRecording()
{
    recorder()->stopRecording ();
}


bool MicrophoneRecorderDesc::
        isStopped()
{
    recorder()->isStopped ();
}


bool MicrophoneRecorderDesc::
        canRecord()
{
    recorder()->canRecord ();
}


void MicrophoneRecorderDesc::
        setDataCallback( IGotDataCallback::Ptr invalidator )
{
    std::vector<Signal::pOperation> sinks;

    if (invalidator) {
        Signal::pOperation marshal(new MarshallNewlyRecordedData(invalidator));
        sinks.push_back (marshal);
    }

    recorder()->getPostSink()->sinks (sinks);
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
    Signal::OperationDesc::Ptr r(new MicrophoneRecorderDesc(input_device_, invalidator_));
    return r;
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
    MicrophoneRecorderDesc::Extent x;
    x.interval = Signal::Interval(0, recorder()->number_of_samples());
    x.number_of_channels = recorder()->num_channels ();
    x.sample_rate = recorder()->sample_rate ();
    return x;
}


MicrophoneRecorder* MicrophoneRecorderDesc::
        recorder() const
{
    return dynamic_cast<MicrophoneRecorder*>(recorder_.get ());
}


} // namespace Adapters


#include "tools/support/timer.h"

#include <QSemaphore>


namespace Adapters {

class GotDataCallback: public MicrophoneRecorderDesc::IGotDataCallback
{
public:
    Signal::Intervals marked_data() const { return marked_data_; }

    virtual void markNewlyRecordedData(Signal::Interval what) {
        marked_data_ |= what;
        semaphore_.release ();
    }

    void wait(int ms_timeout) volatile {
        const_cast<GotDataCallback*>(this)->semaphore_.tryAcquire (1, ms_timeout);
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
        MicrophoneRecorderDesc::IGotDataCallback::Ptr callback(new GotDataCallback);

        MicrophoneRecorderDesc mrd(inputDevice, callback);

        EXCEPTION_ASSERT( mrd.canRecord() );
        EXCEPTION_ASSERT( mrd.isStopped() );

        mrd.startRecording();

        EXCEPTION_ASSERT( !mrd.isStopped() );

        Tools::Support::Timer t;
        dynamic_cast<volatile GotDataCallback*>(callback.get ())->wait (400);
        EXCEPTION_ASSERT_LESS( t.elapsed (), 0.300 );

        mrd.stopRecording();

        EXCEPTION_ASSERT( mrd.isStopped() );

        EXCEPTION_ASSERT(dynamic_cast<const GotDataCallback*>(&*read1(callback))->marked_data () != Signal::Intervals());
        EXCEPTION_ASSERT_LESS( t.elapsed (), 0.400 );
    }
}

} // namespace Adapters
