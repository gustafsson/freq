#include "envelope.h"

#include "tfr/stftdesc.h"
#include "tfr/stftfilter.h"
#include "tfr/stft.h"
#include "tfr/complexbuffer.h"
#include "signal/computingengine.h"

#include "exceptionassert.h"

using namespace Tfr;


namespace Filters {


void Envelope::
        operator()( ChunkAndInverse& chunk )
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


    Tfr::Stft* stft = dynamic_cast<Tfr::Stft*>(chunk.t.get ());
    Tfr::ComplexBuffer::Ptr b = stft->inverseKeepComplex (chunk.chunk);
    Tfr::ChunkElement* d = b->complex_waveform_data ()->getCpuMemory ();
    for (int i=0; i<b->number_of_samples (); ++i)
        d[i] = ChunkElement( abs(d[i]), 0 );

    chunk.inverse = b->get_real ();
}


Tfr::pChunkFilter EnvelopeKernelDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    if (engine == 0 || dynamic_cast<Signal::ComputingCpu*>(engine))
        return Tfr::pChunkFilter(new Envelope);
    return Tfr::pChunkFilter();
}


EnvelopeDesc::
        EnvelopeDesc()
    :
      FilterDesc(Tfr::pTransformDesc(), Tfr::FilterKernelDesc::Ptr(new EnvelopeKernelDesc))
{
    StftDesc* desc;
    Tfr::pTransformDesc t(desc = new StftDesc);
    desc->setWindow (StftDesc::WindowType_Gaussian, 0.75);
    desc->set_approximate_chunk_size (256);
    transformDesc( t );
}



void EnvelopeDesc::
        transformDesc( Tfr::pTransformDesc m )
{
    const StftDesc* desc = dynamic_cast<const StftDesc*>(m.get ());

    EXCEPTION_ASSERT(desc);

    // Ensure redundant transform
    if (!desc->compute_redundant ())
    {
        m = desc->copy ();
        StftDesc* p2 = dynamic_cast<StftDesc*>(m.get ());
        p2->compute_redundant ( true );
    }

    FilterDesc::transformDesc (m);
}

}


#include "expectexception.h"
#include "test/randombuffer.h"


namespace Filters {

void EnvelopeDesc::
        test()
{
    // It should compute the envelope of a signal.
    {
        EnvelopeDesc ed;

        Signal::Operation::Ptr o = ed.createOperation ();

//        _window_size = 256
//        _compute_redundant = true
//        _averaging = 1
//        _enable_inverse = true
//        _overlap = 0.75
//        _window_type = WindowType_Gaussian

        Signal::Interval I(10,12);
        Signal::Interval expected;
        Signal::Interval i = ed.requiredInterval (I, &expected);
        EXCEPTION_ASSERT_EQUALS(expected, Signal::Interval(0,64));
        EXCEPTION_ASSERT_EQUALS(i, Signal::Interval(-192,256));
        Signal::pBuffer b = Test::RandomBuffer::randomBuffer (i, 1, 1);
        Signal::pBuffer r = write1(o)->process(b);

        EXCEPTION_ASSERT_EQUALS(r->getInterval (), expected);

        float* p = r->getChannel (0)->waveform_data ()->getCpuMemory ();
        std::cout << std::endl;
        std::cout << std::endl;
        for (unsigned i=0; i<expected.count (); i++) {
            std::cout << p[i] << ", ";
        }
    }

    // It should only accept StftDesc as TransformDesc.
    {
        EnvelopeDesc ed;

        EXPECT_EXCEPTION(ExceptionAssert,ed.transformDesc(Tfr::pTransformDesc()));
    }
}


} // namespace Filters
