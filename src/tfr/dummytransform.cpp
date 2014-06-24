#include "dummytransform.h"

#include "tfr/chunk.h"
#include "test/randombuffer.h"

#include "exceptionassert.h"
#include "neat_math.h"
#include "demangle.h"

namespace Tfr {

TransformDesc::ptr DummyTransformDesc::
        copy() const
{
    return TransformDesc::ptr(new DummyTransformDesc);
}


pTransform DummyTransformDesc::
        createTransform() const
{
    return pTransform(new DummyTransform);
}


float DummyTransformDesc::
        displayedTimeResolution( float FS, float /*hz*/ ) const
{
    return .25f / FS;
}


FreqAxis DummyTransformDesc::
        freqAxis( float FS ) const
{
    return FreqAxis();
}


unsigned DummyTransformDesc::
        next_good_size( unsigned current_valid_samples_per_chunk, float ) const
{
    return std::max(1u, current_valid_samples_per_chunk)*2;
}


unsigned DummyTransformDesc::
        prev_good_size( unsigned current_valid_samples_per_chunk, float ) const
{
    return std::max(1u, current_valid_samples_per_chunk/2);
}


Signal::Interval DummyTransformDesc::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Signal::Interval DummyTransformDesc::
        affectedInterval( const Signal::Interval& I ) const
{
    return I;
}


std::string DummyTransformDesc::
        toString() const
{
    return vartype(*this);
}


bool DummyTransformDesc::
        operator==(const TransformDesc& b) const
{
    return typeid(*this) == typeid(b);
}


/// DummyTransform
class DummyChunk: public Tfr::Chunk {
public:
    DummyChunk():Chunk(Order_row_major) {}

    Signal::pMonoBuffer original;
};


const TransformDesc* DummyTransform::
        transformDesc() const
{
    return &desc;
}


pChunk DummyTransform::
        operator()( Signal::pMonoBuffer b )
{
    // This is the dummy part.
    DummyChunk*p;
    pChunk c(p = new DummyChunk);
    p->original = b;

    c->chunk_offset = b->sample_offset ();
    c->first_valid_sample = 0;
    c->n_valid_samples = b->number_of_samples ();
    c->original_sample_rate = b->sample_rate ();
    c->sample_rate = b->sample_rate ();
    c->transform_data.reset (new Tfr::ChunkData(b->waveform_data ()->size ()));

    return c;
}


Signal::pMonoBuffer DummyTransform::
        inverse( pChunk chunk )
{
    DummyChunk*p = dynamic_cast<DummyChunk*>(chunk.get ());
    EXCEPTION_ASSERT(p);

    return p->original;
}

} // namespace Tfr

#include "timer.h"
#include "trace_perf.h"

namespace Tfr {

void DummyTransformDesc::
        test ()
{
    // It should describe a transform that doesn't compute anything.
    {
        DummyTransformDesc d;
        pTransform t = d.createTransform ();
        EXCEPTION_ASSERT(t);

        Signal::pMonoBuffer m = Test::RandomBuffer::smallBuffer()->getChannel (0);
        pChunk c = (*t)(m);
        EXCEPTION_ASSERT(c);
    }
}


void DummyTransform::
        test ()
{
    // It should transform a buffer into a dummy state and back.
    for (int i=0; i<2; i++) {
        TRACE_PERF(i==0
                   ? "It should init dummytransform"
                   : "It should transform a buffer into a dummy state and back");

        DummyTransform t;

        Signal::pBuffer b = Test::RandomBuffer::smallBuffer(std::hash<std::string>()("DummyTransform"));
        pChunk c = t(b->getChannel (0));
        EXCEPTION_ASSERT(c);

        Signal::pMonoBuffer b2 = t.inverse (c);
        EXCEPTION_ASSERT(b->getChannel (0) == b2);

        EXCEPTION_ASSERT_EQUALS(c->getCoveredInterval (), b->getInterval ());
    }
}

} // namespace Tfr
