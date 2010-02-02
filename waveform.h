#ifndef WAVEFORM_H
#define WAVEFORM_H

#include <boost/scoped_ptr.hpp>
#include <GpuCpuData.h>

namespace audiere
{
    class SampleSource;
}

class Waveform
{
public:

    Waveform();

    /**
      Reads an audio file using libaudiere
      */
    Waveform(const char* filename);

    /**
      Writes wave audio with 16 bits per sample
      */
    void writeFile( const char* filename );
    void play() const;

    int _sample_rate;
    int channel_count() {
        return _waveformData->getNumberOfElements().height;
    }

    boost::scoped_ptr<GpuCpuData<float> > _waveformData;

protected:
    audiere::SampleSource* _source;
    std::string _last_filename;
};

#endif // WAVEFORM_H
