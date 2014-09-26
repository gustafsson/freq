#include "flacfile.h"
#include "cpumemorystorage.h"
#include "log.h"
#include <limits>
#include <flac/all.h>

using namespace Signal;

template<class T>
void transpose(const pBuffer& out, const void* vin)
{
    const T* in = (const T*)vin;
    int C = out->number_of_channels ();
    int L = out->number_of_samples ();

    for (int c=0; c<C; c++)
    {
        pMonoBuffer m = out->getChannel (c);
        float* p = CpuMemoryStorage::WriteAll<1>(m->waveform_data()).ptr ();

        for (int s=0; s<L; s++)
            p[s] = in[s*C+c];
    }
}


struct FlacFormat {
    pBuffer block;
    FLAC__uint64 total_samples = 0;
    unsigned sample_rate = 0;
    unsigned channels = 0;
    unsigned bps = 0;
    unsigned last_blocksize = 0;
};


class FileReader : public Operation
{
public:
    FileReader(FLAC__StreamDecoder* decoder, FlacFormat*const fmt)
        :
          decoder(decoder),
          fmt(fmt)
    {
    }

    pBuffer process(pBuffer b) override
    {
        Interval I = b->getInterval ();

        Signal::IntervalType i = std::max(0ll, I.first);
        FLAC__stream_decoder_seek_absolute(decoder, i);
        while (true)
        {
            fmt->block->set_sample_offset (i);
            *b |= *fmt->block;
            i += fmt->last_blocksize;

            if (i>=I.last)
                break;

            if (!FLAC__stream_decoder_process_single (decoder))
            {
                Log("flacfile: failed reading at %d") % i;
                break;
            }
        }

        // if reading before start, zeros
        if (I.first < 0)
            *b |= Buffer(Signal::Interval(I.first, 0), b->sample_rate (), b->number_of_channels ());

        // if b extends EOF, zeros
        if (i < I.last)
            *b |= Buffer(Interval(i,I.last), b->sample_rate (), b->number_of_channels ());

        return b;
    }

    FLAC__StreamDecoder* decoder;
    FlacFormat* fmt;
};


static FLAC__StreamDecoderWriteStatus write_callback(const FLAC__StreamDecoder *decoder, const FLAC__Frame *frame, const FLAC__int32 * const buffer[], void *client_data);
static void metadata_callback(const FLAC__StreamDecoder *decoder, const FLAC__StreamMetadata *metadata, void *client_data);
static void error_callback(const FLAC__StreamDecoder *decoder, FLAC__StreamDecoderErrorStatus status, void *client_data);

FLAC__StreamDecoderWriteStatus write_callback(const FLAC__StreamDecoder *decoder, const FLAC__Frame *frame, const FLAC__int32 * const buffer[], void *client_data)
{
    FlacFormat *f = (FlacFormat*)client_data;
    unsigned i, c;

    (void)decoder;

    if(f->total_samples == 0) {
        Log("flacfile: Freq only supports FLAC files that have a total_samples count in STREAMINFO");
        return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
    }

    if (!f->block || f->block->number_of_samples() < frame->header.blocksize)
        f->block.reset(new Buffer(0, frame->header.blocksize, f->sample_rate, f->channels));

    float n = 1ll << f->bps;
    float a = -n/2;

    for (c = 0; c < f->channels; c++)
    {
        pMonoBuffer m = f->block->getChannel (c);
        float* p = CpuMemoryStorage::WriteAll<1>(m->waveform_data()).ptr ();

        for (i = 0; i < frame->header.blocksize; i++)
        {
            FLAC__int32 v = buffer[c][i];
            p[i] = -1.f + 2.f * (v-a)/n;
        }
    }

    f->last_blocksize = frame->header.blocksize;

    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}

void metadata_callback(const FLAC__StreamDecoder *decoder, const FLAC__StreamMetadata *metadata, void *client_data)
{
    FlacFormat *f = (FlacFormat*)client_data;

    (void)decoder;

    /* print some stats */
    if(metadata->type == FLAC__METADATA_TYPE_STREAMINFO) {
        /* save for later */
        f->total_samples = metadata->data.stream_info.total_samples;
        f->sample_rate = metadata->data.stream_info.sample_rate;
        f->channels = metadata->data.stream_info.channels;
        f->bps = metadata->data.stream_info.bits_per_sample;
        Log("flacfile: reading file. N=%d. fs=%d. C=%d. bps=%d")
                % f->total_samples % f->sample_rate % f->channels % f->bps;
    }
    else
    {
        Log("flacfile: metadata->type = %d") % metadata->type;
    }
}

void error_callback(const FLAC__StreamDecoder *decoder, FLAC__StreamDecoderErrorStatus status, void *client_data)
{
    (void)decoder, (void)client_data;

    Log("flacfile: error callback: %s") % FLAC__StreamDecoderErrorStatusString[status];
}


FlacFile::FlacFile(QUrl url)
    :   url(url),
        decoderp(0),
        fmt(new FlacFormat)
{
    // TODO read this one bit at a time, upon request, and don't keep anything in memory
    FLAC__StreamDecoderInitStatus init_status;
    bool ok = false;

    // https://git.xiph.org/?p=flac.git;a=blob_plain;f=examples/c/decode/file/main.c;hb=HEAD
    if((decoderp = FLAC__stream_decoder_new()) == NULL) {
        Log("flacfile: ERROR allocating decoder");
        return;
    }
	
    FLAC__StreamDecoder *decoder = (FLAC__StreamDecoder*)decoderp;
    (void)FLAC__stream_decoder_set_md5_checking(decoder, true);

    init_status = FLAC__stream_decoder_init_file(
                decoder, url.toLocalFile ().toStdString ().c_str (),
                write_callback, metadata_callback, error_callback, /*client_data=*/fmt);

    if(init_status != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
        Log("flacfile: ERROR initializing decoder %s when reading %s")
                % FLAC__StreamDecoderInitStatusString[init_status]
                % url.toLocalFile ().toStdString ();
        FLAC__stream_decoder_delete(decoder);
        decoderp = 0;
        return;
    }

    ok = FLAC__stream_decoder_process_until_end_of_metadata(decoder);
    Log("flacfile: decoding: %s\n") % (ok? "succeeded" : "FAILED");
    Log("flacfile:    state: %s\n") % FLAC__StreamDecoderStateString[FLAC__stream_decoder_get_state(decoder)];
}


FlacFile::
        ~FlacFile()
{
    if (FLAC__StreamDecoder *decoder = (FLAC__StreamDecoder*)decoderp)
        FLAC__stream_decoder_delete(decoder);

    delete fmt;
}


Interval FlacFile::
        requiredInterval( const Interval& I, Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Interval FlacFile::
        affectedInterval( const Interval& I ) const
{
    return I;
}


OperationDesc::ptr FlacFile::
        copy() const
{
    return OperationDesc::ptr();
}


Operation::ptr FlacFile::
        createOperation(ComputingEngine* engine) const
{
    if (engine)
        return Operation::ptr();

    return Operation::ptr(new FileReader((FLAC__StreamDecoder*)decoderp, fmt));
}


OperationDesc::Extent FlacFile::
        extent() const
{
    if (!fmt->total_samples)
        return Extent();

    Extent x;
    x.interval = Interval(0,fmt->total_samples);
    x.number_of_channels = fmt->channels;
    x.sample_rate = fmt->sample_rate;
    return x;
}


QString FlacFile::
        toString() const
{
    return QString("FlacFile %1%").arg(url.toString ());
}


bool FlacFile::
        operator==(const OperationDesc& d) const
{
    const FlacFile* b = dynamic_cast<const FlacFile*>(&d);
    return b && b->url == url;
}
