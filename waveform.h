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
    /**
      Reads an audio file using libaudiere
      */
    Waveform(const char* filename);

    Waveform() {}

    /**
      Writes wave audio with 16 bits per sample
      */
    void writeFile( const char* filename ) const;

    int _sample_rate;
    int channel_count() {
        return _waveformData->getNumberOfElements().height;
    }

    boost::scoped_ptr<GpuCpuData<float> > _waveformData;

protected:
    audiere::SampleSource* _source;
};

#endif // WAVEFORM_H
