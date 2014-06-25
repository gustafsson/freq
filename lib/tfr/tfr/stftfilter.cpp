#include "stftfilter.h"
#include "stft.h"

#include "signal/computingengine.h"

#include "neat_math.h"
#include <memory.h>

//#define TIME_StftFilter
#define TIME_StftFilter if(0)

using namespace Signal;

namespace Tfr {

StftFilterDesc::
        StftFilterDesc()
{
    StftDesc* desc;
    Tfr::pTransformDesc t(desc = new StftDesc);
    desc->setWindow(Tfr::StftDesc::WindowType_Hann, 0.75f);
    transformDesc( t );
}


void StftFilterDesc::
        transformDesc( Tfr::pTransformDesc m )
{
    const StftDesc* desc = dynamic_cast<const StftDesc*>(m.get ());

    EXCEPTION_ASSERT(desc);

    ChunkFilterDesc::transformDesc(m);
}

} // namespace Signal
