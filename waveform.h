#ifndef WAVEFORM_H
#define WAVEFORM_H

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <GpuCpuData.h>
#include <set>

typedef boost::shared_ptr<class Waveform> pWaveform;
typedef boost::shared_ptr<class Waveform_chunk> pWaveform_chunk;

class Waveform_chunk {
public:
    enum Interleaved {
        Interleaved_Complex,
        Only_Real
    };

    Waveform_chunk(Interleaved interleaved=Only_Real);

    boost::scoped_ptr<GpuCpuData<float> > waveform_data;

    Interleaved interleaved() const {return _interleaved; }
    pWaveform_chunk getInterleaved(Interleaved);

    unsigned sample_offset;
    unsigned sample_rate;
    bool modified;

    std::set<unsigned> valid_transform_chunks;
private:
    const Interleaved _interleaved;
};

namespace audiere
{
    class SampleSource;
}

typedef boost::shared_ptr<class Waveform> pWaveform;

class Waveform
{
public:

    Waveform();
    Waveform(const char* filename);

    pWaveform_chunk getChunk( unsigned firstSample, unsigned numberOfSamples, unsigned channel, Waveform_chunk::Interleaved interleaved );
    void setChunk( pWaveform_chunk chunk ) { _waveform = chunk; }
    pWaveform_chunk getChunkBehind() { return _waveform; }

    /**
      Writes wave audio with 16 bits per sample
      */
    void writeFile( const char* filename );
    pWaveform crop();
    void play();

    unsigned channel_count() {        return _waveform->waveform_data->getNumberOfElements().height; }
    unsigned sample_rate() {          return _waveform->sample_rate;    }
    unsigned number_of_samples() {    return _waveform->waveform_data->getNumberOfElements().width; }
    float length() {             return number_of_samples() / (float)sample_rate(); }

private:
    audiere::SampleSource* _source;

    pWaveform_chunk _waveform;

    std::string _last_filename;
};

#endif // WAVEFORM_H
