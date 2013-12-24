#include "writewav.h"

#include "neat_math.h" // defines __int64_t which is expected by sndfile.h

#include <float.h>

#include <sndfile.hh> // for writing various formats
 
#include "Statistics.h"
#include "cpumemorystorage.h"
#include "signal/transpose.h"
#include "exceptionassert.h"

#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include <QTemporaryFile>


//#define TIME_WRITEWAV
#define TIME_WRITEWAV if(0)

//#define TIME_WRITEWAV_LINE(x) TIME(x)
#define TIME_WRITEWAV_LINE(x) x


using namespace boost;
namespace Adapters {

WriteWav::
        WriteWav( std::string filename )
:   _filename(filename),
    _normalize(false)
{
    TaskInfo("WriteWav %s", _filename.c_str());
    reset();
}


WriteWav::
        ~WriteWav()
{
    TaskInfo tt("~WriteWav %s", _filename.c_str());
    _sndfile.reset();
}


void WriteWav::
        put( Signal::pBuffer buffer )
{
    TIME_WRITEWAV TaskTimer tt("WriteWav::put %s", buffer->getInterval().toString().c_str());

    // Make sure that all of buffer is expected

    if (buffer->getInterval() - _invalid_samples)
    {
        Signal::Interval expected = (buffer->getInterval() & _invalid_samples).spannedInterval();
        EXCEPTION_ASSERT( expected );

        buffer = Signal::BufferSource(buffer).readFixedLength( expected );
    }

    if (!_sndfile)
    {
        const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_16;
        _sndfile.reset(new SndfileHandle(_filename, SFM_WRITE, format, buffer->number_of_channels (), buffer->sample_rate()));

        if (!*_sndfile)
        {
            TaskInfo("ERROR: WriteWav(%s) libsndfile couldn't create a file for %u channels and sample rate %f",
                     _filename.c_str(), buffer->number_of_channels (), buffer->sample_rate());
            return;
        }
    }

    _invalid_samples -= buffer->getInterval();

    buffer->sample_offset();
    appendBuffer(buffer);

    if (!_invalid_samples)
    {
        _sndfile.reset();

        if (_normalize)
            rewriteNormalized();
    }
}


void WriteWav::
        reset()
{
    _sndfile.reset();
    _invalid_samples.clear();
    _sum = 0;
    _sumsamples = 0;
    _offset = 0;
    _low = FLT_MAX;
    _high = -FLT_MAX;
}


void WriteWav::
        invalidate_samples( const Signal::Intervals& s )
{
    _invalid_samples |= s;
    _offset = _invalid_samples.spannedInterval().first;
}


Signal::Intervals WriteWav::
        invalid_samples()
{
    return _invalid_samples;
}


bool WriteWav::
        deleteMe()
{
    // don't delete, in case normalize is changed after the file has been written
    return false;
}


void WriteWav::
        normalize(bool v)
{
    if (_normalize == v)
        return;

    _normalize = v;

    if (!_invalid_samples)
        rewriteNormalized();
}


void WriteWav::
        writeToDisk(std::string filename, Signal::pBuffer b, bool normalize)
{
    WriteWav* w = 0;
    Signal::Operation::Ptr writer(w = new WriteWav(filename));
    Signal::Operation::WritePtr ww(writer);
    w->normalize( normalize );

    w->invalidate_samples(b->getInterval());
    w->put(b);
}


void WriteWav::
        appendBuffer(Signal::pBuffer b)
{
    TIME_WRITEWAV TaskTimer tt("%s %s %s", __FUNCTION__, _filename.c_str(),
                               b->getInterval().toString().c_str());

    EXCEPTION_ASSERT( _sndfile );
    EXCEPTION_ASSERT( *_sndfile );

    DataStorage<float> interleaved_data(b->number_of_channels (), b->number_of_samples());
    TIME_WRITEWAV_LINE(Signal::transpose( &interleaved_data, b->mergeChannelData().get() ));

    double sum = 0;
    float high = _high;
    float low = _low;

    float* p = CpuMemoryStorage::ReadOnly<float,2>( &interleaved_data ).ptr();
    int N = b->number_of_channels () * b->number_of_samples();

    {
    TaskTimer ta("sum/high/low");
    for (int i=0; i<N; ++i)
    {
        const float v = p[i];
        sum += v;
        high = std::max(high, v);
        low = std::min(low, v);
    }
    }

    _high = high;
    _low = low;
    _sum += sum;
    _sumsamples += N;

    _sndfile->seek((b->sample_offset() - _offset).asInteger(), SEEK_SET);
    TIME_WRITEWAV_LINE(_sndfile->write( p, N )); // sndfile will convert float to short int
}


void WriteWav::
        rewriteNormalized()
{
    _sndfile.reset();

    std::string tempfilename;
    {
        QTemporaryFile tempname("XXXXXX.wav");

        tempname.open();

        tempfilename = tempname.fileName().toStdString();
    }

    bool renamestatus = QFile::rename( _filename.c_str(), tempfilename.c_str() );

    TaskInfo ti("renaming '%s' to '%s': %s",
                _filename.c_str(), tempfilename.c_str(),
                renamestatus?"success":"failed");
    if (!renamestatus)
        return;

    try
    {
        SndfileHandle inputfile(tempfilename);
        if (!inputfile || 0 == inputfile.frames())
        {
            TaskInfo("ERROR: Couldn't read from %s", tempfilename.c_str());
            return;
        }
        SndfileHandle outputfile(_filename, SFM_WRITE, inputfile.format(), inputfile.channels(), inputfile.samplerate());
        if (!outputfile)
        {
            TaskInfo("ERROR: Couldn't write to %s", _filename.c_str());
            return;
        }

        long double mean = _sum/_sumsamples;

        //    -1 + 2*(v - low)/(high-low);
        //    -1 + (v - low)/std::max(_high-mean, mean-_low)

        long double k = std::max(_high-mean, mean-_low);
        float affine_s = 1/k;
        float affine_d = -1 - _low/k;

        if (!_normalize)
        {
            affine_d = mean;
            affine_s = k;
        }

        // when sndfile converts float to 16-bit integers it doesn't bother with rounding to nearest. Adding an offset in advance accomodates for that.
        affine_d += 1.f/(1<<16);

        sf_count_t frames = inputfile.frames();
        TaskInfo ti2("rewriting %u frames", (unsigned)frames);
        int num_channels = inputfile.channels();
        size_t frames_per_buffer = (4 << 20)/sizeof(float)/num_channels; // 4 MB buffer
        std::vector<float> buffer(num_channels * frames_per_buffer);
        float* p = &buffer[0];

        while (true)
        {
            sf_count_t items_read = inputfile.read(p, buffer.size());
            if (0 == items_read)
                break;

            for (int x=0; x<items_read; ++x)
                p[x] = affine_d + affine_s*p[x];

            outputfile.write(p, items_read );
        }

    } catch (...) {
        QFile::remove(tempfilename.c_str());
        throw;
    }

    QFile::remove(tempfilename.c_str());
}


} // namespace Adapters
