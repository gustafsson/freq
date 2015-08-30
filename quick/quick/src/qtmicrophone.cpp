#include "qtmicrophone.h"
#include "log.h"
#include "cpumemorystorage.h"
#include "heightmap/uncaughtexception.h"
#include "neat_math.h"

#include <QAudioInput>

#ifdef __APPLE__
    #include <TargetConditionals.h>
#endif

//#define SKIP_ZEROS
//#define LOG_DATA
#define LOG_DATA if(0)

// Without QTMICROPHONETHREAD the microphone callback is processed in the main
// thread which may not poll the event queue fast enough. Use a separate thread
// instead so avoid dropping frames if/when the main thread blocks for a
// fraction of a second for whatever reason.
#define QTMICROPHONETHREAD

GotData::GotData(
        shared_state<Signal::Recorder::Data> data,
        Signal::Processing::IInvalidator::ptr& invalidator,
        QAudioFormat format,
        QObject* parent)
    :
      QIODevice(parent),
      data(data),
      invalidator(invalidator),
      format(format)
{
    open(QIODevice::WriteOnly);
}


GotData::~GotData() {
    close();
}


qint64 GotData::
        readData(char *data, qint64 maxlen)
{
    Q_UNUSED(data)
    Q_UNUSED(maxlen)

    return 0;
}


template<class T>
void makefloat(const char* data, qint64 len, std::vector<float>& f)
{
    T* p = (T*)data;
    qint64 tlen = len / sizeof(T);
    f.resize (tlen);
    float minval = std::numeric_limits<T>::min ();
    float maxval = std::numeric_limits<T>::max ();
    for (qint64 i=0; i < tlen; i++)
        f[i] = -1.f + 2.f*(p[i] - minval)/(1.f + maxval - minval);
}

qint64 GotData::
        writeData(const char *in, qint64 len)
{
    switch (format.sampleType ())
    {
    case QAudioFormat::Float:
        writeData((const float*)in, len/format.channelCount ()/sizeof(float));
        break;
    case QAudioFormat::SignedInt:
    case QAudioFormat::UnSignedInt:
        switch (format.sampleSize ())
        {
        case 8:
            if (format.sampleType () == QAudioFormat::SignedInt) makefloat<qint8>(in, len, f);
            if (format.sampleType () == QAudioFormat::UnSignedInt) makefloat<quint8>(in, len, f);
            break;
        case 16:
            if (format.sampleType () == QAudioFormat::SignedInt) makefloat<qint16>(in, len, f);
            if (format.sampleType () == QAudioFormat::UnSignedInt) makefloat<quint16>(in, len, f);
            break;
        case 32:
            if (format.sampleType () == QAudioFormat::SignedInt) makefloat<qint32>(in, len, f);
            if (format.sampleType () == QAudioFormat::UnSignedInt) makefloat<quint32>(in, len, f);
            break;
        default:
            Log("Unknown sample size %d") % format.sampleSize ();
            return 0;
        }
        writeData(&f[0], f.size ());
        break;
    default:
        Log("Unknown sample type %d") % format.sampleType ();
        return 0;
    }

    return len;
}


void GotData::
       writeData(const float* in, quint64 len)
{
    LOG_DATA Log("qtmicrophone: got %d samples") % len;

    // Prepare buffer
    DataAccessPosition_t number_of_samples = DataAccessPosition_t(len / data.raw()->num_channels);
    if (!buffer || buffer->number_of_samples () != number_of_samples)
        buffer.reset (new Signal::Buffer(0, number_of_samples, data.raw()->sample_rate, data.raw()->num_channels));

    // get pointers to intermediate buffers
    unsigned k = 0;
    unsigned C = buffer->number_of_channels ();
    float* P[C], v[C];
    bool prev_nonzero = true;

    for (unsigned i=0; i<C; ++i)
    {
        Signal::pMonoBuffer b = buffer->getChannel (i);
        P[i] = CpuMemoryStorage::WriteAll<1>(b->waveform_data()).ptr ();
    }

    // transpose and skip zeros
    for (DataAccessPosition_t j=0; j<number_of_samples; ++j)
    {
#ifdef SKIP_ZEROS
        bool nonzero = false;
#else
        bool nonzero = true;
#endif
        for (unsigned i=0; i<C; ++i)
        {
            v[i] = in[j*C + i];
            if (fabs(v[i]) > 1.f/1024.f)
                nonzero = true;
        }

        if (k==1 && !nonzero)
            k = 0;

        if (nonzero || prev_nonzero)
        {
            for (unsigned i=0; i<C; ++i)
                P[i][k] = v[i];
            k++;
        }

        prev_nonzero = nonzero;
    }

    if (k<=10)
        return;

    // Publish
    auto dw = data.write ();
    Signal::IntervalType start = dw->samples.spannedInterval().last;
    buffer->set_sample_offset (start);
    dw->samples.put( buffer ); // copy

    // Ignore samples not written to
    Signal::Interval I(start, start+k);
    dw->samples.invalidate_samples( Signal::Interval(I.last, Signal::Interval::IntervalType_MAX));
    dw.unlock ();

    Signal::Processing::IInvalidator::ptr inv = invalidator;
    if (inv) inv->deprecateCache ( I );
}


template<class T> T max(const QList<T>& V) {
    T v = std::numeric_limits<T>::min ();
    for (const auto& t : V) v = std::max(v,t);
    return v;
}


template<class T> T logclosest(const QList<T>& V, T v) {
    T r = std::numeric_limits<T>::max ();
    for (const auto& t : V)
        if (fabs(log(t)-log(v)) < fabs(log(v)-log(r)))
            r = t;
    return r;
}


QtAudioObject::
        QtAudioObject(QAudioDeviceInfo info, QAudioFormat format, QIODevice* device)
    :
      info_(info),
      format_(format),
      device_(device)
{
    device->setParent (this);
}


QtAudioObject::
        ~QtAudioObject()
{
}


void QtAudioObject::
        startRecording()
{
    if (QThread::currentThread () != this->thread ())
      {
        // Dispatch
       QMetaObject::invokeMethod (this, "startRecording");
        return;
      }

    audio_->start (device_);

    if (audio_->error () != QAudio::NoError)
    {
        Log("QtMicrophone::startRecording failed: %d") % audio_->error ();
        return;
    }
}


void QtAudioObject::
        stopRecording()
{
    if (QThread::currentThread () != this->thread ())
      {
        // Dispatch
       QMetaObject::invokeMethod (this, "stopRecording");
        return;
      }

    audio_->stop ();
}


bool QtAudioObject::
        isStopped()
{
    switch(audio_->state ())
    {
    case QAudio::StoppedState:
    case QAudio::SuspendedState:
        return true;
    default:
        return false;
    }
}


bool QtAudioObject::
        canRecord()
{
    return QAudio::NoError == audio_->error ();
}


void QtAudioObject::
        init()
{
    audio_ = new QAudioInput(info_, format_, this);
#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
    audio_->setBufferSize (1<<12); // 4096, buffer_size/sample_rate = latency -> 93 ms
    //    audio_->setBufferSize (1<<9); // 512, buffer_size/sample_rate = latency -> 12 ms
#else
    #ifdef _DEBUG
        audio_->setBufferSize ( lpo2s(format_.sampleRate ()/10) ); // latency -> 1/10 s
    #else
        audio_->setBufferSize ( lpo2s(format_.sampleRate ()/60/2) ); // latency -> 1/120 s
    #endif
#endif

    Log("qtmicrophone: fs = %d, bits = %d, %d channels, buffer: %d samples")
            % format_.sampleRate () % format_.sampleSize ()
            % format_.channelCount () % audio_->bufferSize ();

    auto e = audio_->error ();
    if (e != QAudio::NoError)
        Log("QtMicrophone error: ") % (int)e;
}


void QtAudioObject::
        finished()
{
#ifndef QTMICROPHONETHREAD
    // Drop any recorded data not already processed (don't stopRecording()
    // as it would invoke GotData::writeData which uses invalidator which
    // in turn uses the dag.

    // The Dag is locked by the caller of the destructor so the invalidator would cause a deadlock.
#endif

    if (audio_)
    {
        audio_->suspend();
        audio_->reset();
        audio_->stop();
        audio_ = 0;
    }
}


QtMicrophone::
        QtMicrophone(QObject* threadOwner)
{
    qRegisterMetaType<QAudio::State>("QAudio::State");
    QAudioDeviceInfo info = QAudioDeviceInfo::defaultInputDevice();
    QAudioFormat format = info.preferredFormat ();
    format.setSampleRate (logclosest(info.supportedSampleRates (),44100));
//    format.setSampleType (QAudioFormat::Float);
    format.setSampleSize (std::min(32,max(info.supportedSampleSizes ())));
    format.setChannelCount (1);

    if (!info.isFormatSupported(format))
        format = info.nearestFormat(format);

    _data.reset (new Signal::Recorder::Data(format.sampleRate (), format.channelCount ()));
    QIODevice* device = new GotData(_data, _invalidator, format);

    audioobject_ = new QtAudioObject(info, format, device);

#ifdef QTMICROPHONETHREAD
    audiothread_ = new QThread;
    // When threadOwner is destroyed destroy the audio object -> when done stop the thread -> when done destroy the thread object
    QObject::connect (threadOwner, SIGNAL(destroyed(QObject*)), audioobject_, SLOT(deleteLater()));
    QObject::connect (audioobject_, SIGNAL(destroyed(QObject*)), audiothread_, SLOT(quit()));
    QObject::connect (audiothread_, SIGNAL(finished()), audiothread_, SLOT(deleteLater()));
    audioobject_->moveToThread (audiothread_);
    audiothread_->start ();
    // Dispatch
    QMetaObject::invokeMethod (audioobject_, "init", Qt::BlockingQueuedConnection);
#else
    audioobject_->init ();
#endif
}


QtMicrophone::
        ~QtMicrophone()
{
#ifdef QTMICROPHONETHREAD
    _invalidator.reset ();
    // Can't wait for audiothread_ to finish becuase it might be using the Dag
    // which is locked by the caller of this destructor but audiothread_ will
    // delete itself when finished
    if (audiothread_)
    {
        audiothread_->quit ();
        audiothread_->deleteLater ();
    }
#else
    audioobject_->finished ();
    delete audioobject_;
#endif
}


void QtMicrophone::
        startRecording()
{
    _offset = length();
    _start_recording.restart ();
    if (audioobject_) audioobject_->startRecording ();
}


void QtMicrophone::
        stopRecording()
{
    if (audioobject_) audioobject_->stopRecording ();
}


bool QtMicrophone::
        isStopped() const
{
    if (!audioobject_)
        return true;
    return audioobject_->isStopped();
}


bool QtMicrophone::
        canRecord()
{
    if (!audioobject_)
        return false;
    return audioobject_->isStopped ();
}


std::string QtMicrophone::
        name()
{
    return "Microphone";
}
