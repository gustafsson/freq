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
 * than 256 samples. Or a frequency lower than FS/256 (i.e 44100/256=172 Hz).
 * Note that a lower modulation frequency is still detected.
 */
class Envelope: public Tfr::ChunkFilter
{
public:
    bool operator()( Tfr::ChunkAndInverse& chunk );
};


class EnvelopeKernelDesc: public Tfr::FilterKernelDesc
{
public:
    Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* engine) const;
};


/**
 * @brief The EnvelopeDesc class should compute the envelope of a signal.
 *
 * It should only accept StftDesc as TransformDesc.
 */
class EnvelopeDesc: public Tfr::FilterDesc
{
public:
    EnvelopeDesc();

    void transformDesc( Tfr::pTransformDesc m );

public:
    static void test();
};

} // namespace Filters

#endif // FILTERS_ENVELOPE_H
