#ifndef TFRCWT_H
#define TFRCWT_H

#include "signal-source.h"
#include "tfr-stft.h"

namespace Tfr {

class Cwt
{
public:
    Cwt( cudaStream_t stream=0 );

    pChunk operator()( Signal::pBuffer );

    float     min_hz() const { return _min_hz; }
    void      min_hz(float f);
    float     max_hz(unsigned sample_rate) const { return sample_rate/2; }
    float     number_of_octaves( unsigned sample_rate ) const;
    unsigned  nScales() { return (unsigned)(number_of_octaves() * scales_per_octave()); }
    unsigned  scales_per_octave() const { return _scales_per_octave; }
    void      scales_per_octave( unsigned );

private:
    Stft        _stft;
    cudaStream_t _stream;
    cufftHandle _fft_many;
    float       _min_hz;
    unsigned    _scales_per_octave;
    pChunk      _intermediate_wt;

    void gc();
};
typedef boost::shared_ptr<Cwt> pCwt;

} // namespace Tfr

#endif // TFRCWT_H
