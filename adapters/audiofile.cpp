#ifdef _MSC_VER
typedef __int64 __int64_t;
#else
#include <stdint.h> // defines __int64_t which is expected by sndfile.h
#endif

#include "audiofile.h"
#include "Statistics.h" // to play around for debugging

#include <sndfile.hh> // for reading various formats
#include <math.h>
#include <stdexcept>
#include <iostream>
#include <sstream>

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/algorithm/string.hpp>

#if LEKA_FFT
#include <cufft.h>
#endif

using namespace std;

namespace Adapters {

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
{
    load(filename);
}

void Audiofile::
        load(std::string filename )
{
    _original_filename = filename;

    TaskTimer tt("Loading %s (this=%p)", filename.c_str(), this);

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
    source.read(completeFile.getCpuMemory(), source.channels()*source.frames()); // yes float
    float* data = completeFile.getCpuMemory();

	Signal::pBuffer waveform( new Signal::Buffer(0, source.frames(), source.samplerate(), source.channels()));
    float* target = waveform->waveform_data()->getCpuMemory();

    // Compute transpose of signal
    for (unsigned i=0; i<source.frames(); i++) {
        for (int c=0; c<source.channels(); c++) {
            target[i + c*source.frames()] = data[i*source.channels() + c];
        }
    }

	setBuffer( waveform );

    //_waveform = getChunk( 0, number_of_samples(), 0, Waveform_chunk::Only_Real );
    //Statistics<float> waveform( _waveform->waveform_data.get() );

    tt << "Signal length: " << lengthLongFormat();

    tt.flushStream();

    tt.info("Data size: %lu samples, %lu channels", (size_t)source.frames(), (size_t)source.channels() );
    tt.info("Sample rate: %lu samples/second", source.samplerate() );
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

} // namespace Adapters
