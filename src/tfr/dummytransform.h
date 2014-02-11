#ifndef TFR_DUMMYTRANSFORM_H
#define TFR_DUMMYTRANSFORM_H

#include "transform.h"

namespace Tfr {

/**
 * @brief The DummyTransformDesc class should describe a transform that doesn't compute anything.
 */
class DummyTransformDesc: public TransformDesc
{
public:
    TransformDesc::Ptr copy() const;
    pTransform createTransform() const;
    float displayedTimeResolution( float FS, float hz ) const;
    FreqAxis freqAxis( float FS ) const;
    unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;
    unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;
    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    std::string toString() const;
    bool operator==(const TransformDesc&) const;

public:
    static void test();
};


/**
 * @brief The DummyTransform class should transform a buffer into a dummy state and back.
 */
class DummyTransform: public Transform
{
public:
    const TransformDesc* transformDesc() const;
    pChunk operator()( Signal::pMonoBuffer b );
    Signal::pMonoBuffer inverse( pChunk chunk );

private:
    DummyTransformDesc desc;

public:
    static void test();
};

} // namespace Tfr

#endif // TFR_DUMMYTRANSFORM_H
