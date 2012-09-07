#ifndef CEPSTRUM_H
#define CEPSTRUM_H

#include "transform.h"
#include "stftparams.h"

namespace Tfr {

class Stft;

class CepstrumParams : public StftParams
{
public:
    pTransform createTransform() const;
    virtual FreqAxis freqAxis( float FS ) const;
};


class Cepstrum : public Tfr::Transform
{
public:
    Cepstrum(const CepstrumParams& p = CepstrumParams());

    CepstrumParams params() const { return p; }
    virtual const TransformParams* transformParams() const { return &p; }

    virtual pChunk operator()( Signal::pBuffer b );
    virtual Signal::pBuffer inverse( pChunk chunk );

    Stft* stft();

private:
    const CepstrumParams p;
    Tfr::pTransform stft_;
};


} // namespace Tfr

#endif // CEPSTRUM_H
