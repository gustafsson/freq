#include "cepstrumfilter.h"
#include "cepstrum.h"

using namespace Signal;

namespace Tfr {

CepstrumFilterDesc::
        CepstrumFilterDesc()
{
    CepstrumDesc* desc;
    Tfr::pTransformDesc t(desc = new CepstrumDesc);
    desc->setWindow(Tfr::CepstrumDesc::WindowType_Hann, 0.75f);
    transformDesc( t );
}


void CepstrumFilterDesc::
        transformDesc( Tfr::pTransformDesc m )
{
    const CepstrumDesc* desc = dynamic_cast<const CepstrumDesc*>(m.get ());

    EXCEPTION_ASSERT(desc);

    ChunkFilterDesc::transformDesc( m );
}



} // namespace Signal
