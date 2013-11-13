#ifndef FILTERS_ENVELOPE_H
#define FILTERS_ENVELOPE_H

#include "tfr/stftfilter.h"

namespace Filters {

/**
 * @brief The Envelope class computes the envelope of a discrete signal.
 * The envelope is defined as the magnitude of the analytic representation of the source signal.
 * The analytic representation is computed with the Hilbert transform.
 * The Hilbert transform is implented by STFT with a window of size 256.
 * This means that the envelope doesn't detect a carrier with a period larger
 * than 1024 samples. Or a frequency lower than FS/1024 (i.e 44100/256=172 Hz).
 * Note that the modulation frequency does not depend on the frequency of the
 * carrier.
 */
class Envelope: public Tfr::StftFilter
{
public:
    Envelope();

    void transform( Tfr::pTransform m );

    bool applyFilter( Tfr::ChunkAndInverse& chunk );
    virtual bool operator()( Tfr::Chunk& );
};

} // namespace Filters

#endif // FILTERS_ENVELOPE_H
