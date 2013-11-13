#include "envelope.h"

#include "tfr/stftdesc.h"
#include "tfr/stftfilter.h"
#include "tfr/stft.h"
#include "tfr/complexbuffer.h"

#include "exceptionassert.h"

using namespace Tfr;


namespace Filters {


Envelope::Envelope()
{
    StftDesc desc;
    desc.setWindow (StftDesc::WindowType_Gaussian, 0.75);
    desc.set_approximate_chunk_size (256);
    transform( desc.createTransform () );
}


void Envelope::
        transform( Tfr::pTransform m )
{
    const StftDesc* desc = dynamic_cast<const StftDesc*>(m->transformDesc ());

    // Ensure redundant transform
    if (!desc->compute_redundant ())
    {
        StftDesc p2(*desc);
        p2.compute_redundant ( true );
        m = p2.createTransform ();
    }

    StftFilter::transform(m);
}


bool Envelope::
        applyFilter( ChunkAndInverse& chunk )
{
    // CPU memset
    ChunkElement* p = chunk.chunk->transform_data->getCpuMemory ();
    int frequencies = chunk.chunk->nScales ();
    int windows = chunk.chunk->nSamples ();
    for (int y=0; y<windows; ++y)
    {
        for (int x=frequencies/2; x<frequencies; ++x)
        {
            p[y*frequencies + x] = ChunkElement(0, 0);
        }
    }


    Tfr::Stft* stft = dynamic_cast<Tfr::Stft*>(Tfr::StftFilter::transform ().get ());
    Tfr::ComplexBuffer::Ptr b = stft->inverseKeepComplex (chunk.chunk);
    Tfr::ChunkElement* d = b->complex_waveform_data ()->getCpuMemory ();
    for (int i=0; i<b->number_of_samples (); ++i)
        d[i] = ChunkElement( abs(d[i]), 0 );

    chunk.inverse = b->get_real ();

    return false; // don't compute the inverse again
}


bool Envelope::
        operator()( Tfr::Chunk& )
{
    EXCEPTION_ASSERT( false );
    return false;
}


} // namespace Filters
