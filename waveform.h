#ifndef WAVEFORM_H
#define WAVEFORM_H

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <GpuCpuData.h>

typedef boost::shared_ptr<class Waveform> pWaveform;
typedef boost::shared_ptr<class Waveform_chunk> pWaveform_chunk;

class Waveform_chunk {
public:
    enum Interleaved {
        Interleaved_Complex,
        Only_Real
    };

    Waveform_chunk(Interleaved interleaved);

    boost::scoped_ptr<GpuCpuData<float> > data;

    Interleaved interleaved() const {return _interleaved; }
    pWaveform_chunk getInterleaved(Interleaved);
    unsigned sampleOffset;
private:
    const Interleaved _interleaved;
};

namespace audiere
{
    class SampleSource;
}

class Waveform
{
public:

    Waveform();
    Waveform(const char* filename);

    void           writeFile( const char* filename ) const;
    pWaveform_chunk getChunk( unsigned firstSample, unsigned numberOfSamples, int channel=0, bool interleaved=true );

    int channel_count() {        return _waveformData->getNumberOfElements().height; }
    int sample_rate() {          return _sample_rate;    }

private:
    int _sample_rate;
    audiere::SampleSource* _source;
    Waveform_chunk _waveform;
};

#endif // WAVEFORM_H
