#include "cepstrum.h"
#include "stft.h"
#include "stftkernel.h"

#include "signal/buffer.h"

#include "TaskTimer.h"
#include "neat_math.h"

using namespace Signal;
namespace Tfr {


CepstrumParams init(CepstrumParams p)
{
    CepstrumParams r = p;
    r.compute_redundant( true );
    return r;
}

Cepstrum::
        Cepstrum(const CepstrumParams& p)
    :
      p(init(p))
{
    stft_ = p.StftParams::createTransform();
}


pChunk Cepstrum::
        operator()( pBuffer b )
{
    TaskTimer tt("Cepstrum");
    Stft& ft = *stft();
    pChunk cepstra = ft(b);

    ::cepstrumPrepareCepstra( cepstra->transform_data, 4.f/p.chunk_size() );

    ft.compute( cepstra->transform_data, cepstra->transform_data, FftDirection_Forward );
    cepstra->freqAxis = p.freqAxis( cepstra->original_sample_rate );

    TaskInfo("Cepstrum debug. Was %s , returned %s ",
        b->getInterval().toString().c_str(),
        cepstra->getInterval().toString().c_str());

    return cepstra;

}


Signal::pBuffer Cepstrum::
        inverse( pChunk )
{
    throw std::logic_error("Not implemented");
}


Stft* Cepstrum::
        stft()
{
    return dynamic_cast<Stft*>(stft_.get());
}


pTransform CepstrumParams::
        createTransform() const
{
    return pTransform(new Cepstrum(*this));
}


FreqAxis CepstrumParams::
        freqAxis( float FS ) const
{
    FreqAxis fa;
    fa.setQuefrency( FS, chunk_size());
    return fa;
}


} // namespace Tfr
