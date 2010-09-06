#ifdef _MSC_VER
typedef __int64 __int64_t;
#else
#include <stdint.h> // defines __int64_t which is expected by sndfile.h
#endif

#include "signal/audiofile.h"
#include "signal/playback.h"
#include <sndfile.hh> // for reading various formats
#include <math.h>
#include "Statistics.h"
#include <stdexcept>
#include <iostream>
#include <sstream>

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/algorithm/string.hpp>
//#include <QThread>
//#include <QSound>

#if LEKA_FFT
#include <cufft.h>

#endif

using namespace std;

namespace Signal {

static std::string getSupportedFileFormats (bool detailed=false) {
    SF_FORMAT_INFO	info ;
    SF_INFO 		sfinfo ;
    char buffer [128] ;
    int format, major_count, subtype_count, m, s ;
    stringstream ss;

    memset (&sfinfo, 0, sizeof (sfinfo)) ;
    buffer [0] = 0 ;
    sf_command (NULL, SFC_GET_LIB_VERSION, buffer, sizeof (buffer)) ;
    if (strlen (buffer) < 1)
    {	ss << "Could not retrieve sndfile lib version.";
        return ss.str();
    }
    ss << "Version : " << buffer << endl;

    sf_command (NULL, SFC_GET_FORMAT_MAJOR_COUNT, &major_count, sizeof (int)) ;
    sf_command (NULL, SFC_GET_FORMAT_SUBTYPE_COUNT, &subtype_count, sizeof (int)) ;

    sfinfo.channels = 1 ;
    for (m = 0 ; m < major_count ; m++)
    {	info.format = m ;
            sf_command (NULL, SFC_GET_FORMAT_MAJOR, &info, sizeof (info)) ;
            ss << info.name << "  (extension \"" << info.extension << "\")" << endl;

            format = info.format ;

            if(detailed)
            {
                for (s = 0 ; s < subtype_count ; s++)
                {	info.format = s ;
                        sf_command (NULL, SFC_GET_FORMAT_SUBTYPE, &info, sizeof (info)) ;

                        format = (format & SF_FORMAT_TYPEMASK) | info.format ;

                        sfinfo.format = format ;
                        if (sf_format_check (&sfinfo))
                                ss << "   " << info.name << endl;
                } ;
                ss << endl;
            }
    } ;
    ss << endl;

    return ss.str();
}


// static
std::string Audiofile::
        getFileFormatsQtFilter( bool split )
{
    SF_FORMAT_INFO	info ;
    SF_INFO 		sfinfo ;
    char buffer [128] ;

	int major_count, subtype_count, m ;
    stringstream ss;

    memset (&sfinfo, 0, sizeof (sfinfo)) ;
    buffer [0] = 0 ;
    sf_command (NULL, SFC_GET_LIB_VERSION, buffer, sizeof (buffer)) ;
    if (strlen (buffer) < 1)
    {
        return NULL;
    }

    sf_command (NULL, SFC_GET_FORMAT_MAJOR_COUNT, &major_count, sizeof (int)) ;
    sf_command (NULL, SFC_GET_FORMAT_SUBTYPE_COUNT, &subtype_count, sizeof (int)) ;

    sfinfo.channels = 1 ;
    for (m = 0 ; m < major_count ; m++)
    {	info.format = m ;
            sf_command (NULL, SFC_GET_FORMAT_MAJOR, &info, sizeof (info)) ;
            if (split) {
                if (0<m) ss << ";;";
                string name = info.name;
                boost::replace_all(name, "(", "- ");
                boost::erase_all(name, ")");
                ss << name << " (*." << info.extension << " *." << info.name << ")";
            } else {
                if (0<m) ss << " ";
                ss <<"*."<< info.extension << " *." << info.name;
            }
    }

    return ss.str();
}


/**
  Reads an audio file using libsndfile
  */
Audiofile::
        Audiofile(std::string filename)
:   _selected_channel(0),
    _original_filename(filename)
{
    _waveform.reset( new Buffer());

    TaskTimer tt("%s %s",__FUNCTION__,filename.c_str());

    SndfileHandle source(filename);

    if (0==source || 0 == source.frames()) {
        stringstream ss;

        ss << "Couldn't open '" << filename << "'" << endl
           << endl
           << "Supported audio file formats through Sndfile:" << endl
           << getSupportedFileFormats();

        throw std::ios_base::failure(ss.str());
    }

    GpuCpuData<float> completeFile(0, make_uint3( source.channels(), source.frames(), 1));
    _waveform->waveform_data.reset( new GpuCpuData<float>(0, make_uint3( source.frames(), source.channels(), 1)) );
    _waveform->sample_rate = source.samplerate();
    source.read(completeFile.getCpuMemory(), source.channels()*source.frames()); // yes float
    float* target = _waveform->waveform_data->getCpuMemory();
    float* data = completeFile.getCpuMemory();

    for (unsigned i=0; i<source.frames(); i++) {
        for (int c=0; c<source.channels(); c++) {
            target[i + c*source.frames()] = data[i*source.channels() + c];
        }
    }

    //_waveform = getChunk( 0, number_of_samples(), 0, Waveform_chunk::Only_Real );
    //Statistics<float> waveform( _waveform->waveform_data.get() );

#if LEKA_FFT
/* do stupid things: */
    num_frames = 1<<13;
    GpuCpuData<float> ut2(0, make_cudaExtent(3*num_frames*sizeof(float),1,1));
    GpuCpuData<float> ut(0, make_cudaExtent(3*num_frames*sizeof(float),1,1));
//    for (unsigned i=0; i<num_frames/2; i++)
//        printf("\t%g\n",fdata[i]);

    cufftHandle                             _fft_dummy;
    CufftException_SAFE_CALL(cufftPlan1d(&_fft_dummy, num_frames, CUFFT_R2C, 1));
    CufftException_SAFE_CALL(cufftExecR2C(_fft_dummy,
                               (cufftReal *)_waveform.waveform_data->getCudaGlobal().ptr(),
                               (cufftComplex *)ut2.getCudaGlobal().ptr()));
    CufftException_SAFE_CALL(cufftPlan1d(&_fft_dummy, num_frames, CUFFT_C2C, 1));
    /*CufftException_SAFE_CALL(cufftExecC2C(_fft_dummy,
                               (cufftComplex *)_waveform.waveform_data->getCudaGlobal().ptr(),
                               (cufftComplex *)ut2.getCudaGlobal().ptr(),
                               CUFFT_FORWARD));*/
    CufftException_SAFE_CALL(cufftExecC2C(_fft_dummy,
                               (cufftComplex *)ut2.getCudaGlobal().ptr(),
                               (cufftComplex *)ut.getCudaGlobal().ptr(),
                               CUFFT_INVERSE));
    cufftDestroy( _fft_dummy );
    //GpuCpuData<float> d = *_waveform.waveform_data;
    fdata = ut.getCpuMemory();
    printf("num_frames/2=%d\n", num_frames/2);
    for (unsigned i=0; i<num_frames; i++)
        printf("\t%g  \t%g\n",fdata[2*i], fdata[2*i+1]);
#endif
}


pBuffer Audiofile::read( unsigned firstSample, unsigned numberOfSamples ) {
    return  getChunk( firstSample, numberOfSamples, 0, Buffer::Only_Real );
}

/* returns a chunk with numberOfSamples samples. If the requested range exceeds the source signal it is padded with 0. */
pBuffer Audiofile::getChunk( unsigned firstSample, unsigned numberOfSamples, unsigned channel, Buffer::Interleaved interleaved )
{
    if (firstSample+numberOfSamples < firstSample)
        throw std::invalid_argument("Overflow: firstSample+numberOfSamples");

    if (channel >= _waveform->waveform_data->getNumberOfElements().height)
        throw std::invalid_argument("channel >= _waveform.waveform_data->getNumberOfElements().height");

    char m=1+(Buffer::Interleaved_Complex == interleaved);
    char sourcem = 1+(Buffer::Interleaved_Complex == _waveform->interleaved());

    pBuffer chunk( new Buffer( interleaved ));
    chunk->waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent(m*numberOfSamples, 1, 1) ) );
    chunk->sample_rate = sample_rate();
    chunk->sample_offset = firstSample;
    size_t sourceSamples = _waveform->waveform_data->getNumberOfElements().width/sourcem;

    unsigned validSamples;
    if (firstSample > sourceSamples)
        validSamples = 0;
    else if ( firstSample + numberOfSamples > sourceSamples )
        validSamples = sourceSamples - firstSample;
    else // default case
        validSamples = numberOfSamples;

    float *target = chunk->waveform_data->getCpuMemory();
    float *source = _waveform->waveform_data->getCpuMemory()
                  + firstSample*sourcem
                  + channel * _waveform->waveform_data->getNumberOfElements().width;

    bool interleavedSource = Buffer::Interleaved_Complex == _waveform->interleaved();
    for (unsigned i=0; i<validSamples; i++) {
        target[i*m + 0] = source[i*sourcem + 0];
        if (Buffer::Interleaved_Complex == interleaved)
            target[i*m + 1] = interleavedSource ? source[i*sourcem + 1]:0;
    }

    for (unsigned i=validSamples; i<numberOfSamples; i++) {
        target[i*m + 0] = 0;
        if (Buffer::Interleaved_Complex == interleaved)
            target[i*m + 1] = 0;
    }
    return chunk;
}


/**
  Remove zeros from the beginning and end
  */
pSource Audiofile::crop() {
    unsigned num_frames = _waveform->waveform_data->getNumberOfElements().width;
    unsigned channel_count = _waveform->waveform_data->getNumberOfElements().height;
    float *fdata = _waveform->waveform_data->getCpuMemory();
    unsigned firstNonzero = 0;
    unsigned lastNonzero = 0;
    for (unsigned f=0; f<num_frames; f++)
        for (unsigned c=0; c<channel_count; c++)
            if (fdata[f*channel_count + c])
                lastNonzero = f;
            else if (firstNonzero==f)
                firstNonzero = f+1;

    if (firstNonzero > lastNonzero)
        return pSource();

    Audiofile* wf(new Audiofile());
    pSource rwf(wf);
    wf->_waveform->sample_offset = firstNonzero + _waveform->sample_offset;
    wf->_waveform->sample_rate = sample_rate();
    wf->_waveform->waveform_data.reset (new GpuCpuData<float>(0, make_cudaExtent((lastNonzero-firstNonzero+1) , channel_count, 1)));
    float *data = wf->_waveform->waveform_data->getCpuMemory();


    for (unsigned f=firstNonzero; f<=lastNonzero; f++)
        for (unsigned c=0; c<channel_count; c++) {
            float rampup = min(1.f, (f-firstNonzero)/(sample_rate()*0.01f));
            float rampdown = min(1.f, (lastNonzero-f)/(sample_rate()*0.01f));
            rampup = 3*rampup*rampup-2*rampup*rampup*rampup;
            rampdown = 3*rampdown*rampdown-2*rampdown*rampdown*rampdown;
            data[f-firstNonzero + c*num_frames] = 0.5f*rampup*rampdown*fdata[ f + c*num_frames];
        }

    return rwf;
}


unsigned Audiofile::sample_rate() {          return _waveform->sample_rate;    }
long unsigned Audiofile::number_of_samples() {    return _waveform->waveform_data->getNumberOfElements().width; }

} // namespace Signal
