#ifndef CEPSTRUM_H
#define CEPSTRUM_H

#include "transform.h"
#include "stftdesc.h"

namespace Tfr {

class Stft;

class CepstrumDesc : public StftDesc
{
public:
    pTransform createTransform() const;
    virtual FreqAxis freqAxis( float FS ) const;
};


class Cepstrum : public Tfr::Transform
{
public:
    Cepstrum(const CepstrumDesc& p = CepstrumDesc());

    const CepstrumDesc& desc() const { return p; }
    virtual const TransformDesc* transformDesc() const { return &p; }

    virtual pChunk operator()( Signal::pMonoBuffer b );
    virtual Signal::pMonoBuffer inverse( pChunk chunk );

private:
    const CepstrumDesc p;
};


} // namespace Tfr

#endif // CEPSTRUM_H
