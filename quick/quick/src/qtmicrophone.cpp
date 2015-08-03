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


QtMicrophone::
        QtMicrophone()
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

    audio_.reset (new QAudioInput(info, format));
#ifdef TARGET_OS_IPHONE
    audio_->setBufferSize (1<<12); // 4096, buffer_size/sample_rate = latency -> 93 ms
    //    audio_->setBufferSize (1<<9); // 512, buffer_size/sample_rate = latency -> 12 ms
#else
    #ifdef _DEBUG
        audio_->setBufferSize ( lpo2s(format.sampleRate ()/10) ); // latency -> 1/10 s
    #else
        audio_->setBufferSize ( lpo2s(format.sampleRate ()/60/2) ); // latency -> 1/120 s
    #endif
#endif

    Log("qtmicrophone: fs = %d, bits = %d, %d channels, buffer: %d samples")
            % format.sampleRate () % format.sampleSize ()
            % format.channelCount () % audio_->bufferSize ();

    auto e = audio_->error ();
    if (e != QAudio::NoError)
        Log("QtMicrophone error: ") % (int)e;

    _data.reset (new Recorder::Data(format.sampleRate (), format.channelCount ()));
    device_.reset (new GotData(_data, _invalidator, format));
}


QtMicrophone::
        ~QtMicrophone()
{
    // Drop any recorded data not already processed (don't stopRecording()
    // as it would invoke GotData::writeData which uses invalidator which
    // in turn uses the dag. But the Dag is locked by the caller of this
    // destructor.

    // Waiting for recorder to finish
    audio_->suspend();
    audio_->reset();
    audio_->stop();
}


void QtMicrophone::
        startRecording()
{
    audio_->start (device_.data ());

    if (audio_->error () != QAudio::NoError)
    {
        Log("QtMicrophone::startRecording failed: %d") % audio_->error ();
        return;
    }

    _offset = length();
    _start_recording.restart ();
}


void QtMicrophone::
        stopRecording()
{
    audio_->stop ();
}


bool QtMicrophone::
        isStopped() const
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


bool QtMicrophone::
        canRecord()
{
    return QAudio::NoError == audio_->error ();
}


std::string QtMicrophone::
        name()
{
    return "Microphone";
}
