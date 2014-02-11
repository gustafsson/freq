#include "envelope.h"

#include "tfr/stftdesc.h"
#include "tfr/stftfilter.h"
#include "tfr/stft.h"
#include "tfr/complexbuffer.h"
#include "tfr/transformoperation.h"
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


EnvelopeDesc::
        EnvelopeDesc()
{
    StftDesc* desc;
    Tfr::pTransformDesc t(desc = new StftDesc);
    desc->setWindow (StftDesc::WindowType_Gaussian, 0.75);
    desc->set_approximate_chunk_size (256);
    transformDesc( t );
}


Tfr::pChunkFilter EnvelopeDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    if (engine == 0 || dynamic_cast<Signal::ComputingCpu*>(engine))
        return Tfr::pChunkFilter(new Envelope);
    return Tfr::pChunkFilter();
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

    ChunkFilterDesc::transformDesc(m);
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
        ChunkFilterDesc::Ptr cfd(new EnvelopeDesc);
        Tfr::TransformOperationDesc ed(cfd);
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
        float q[] = {3.24838, 5.65101, 4.82303, 3.30121, 3.34537, 6.11119, 6.78169, 4.89376, 2.8429, 0.457352, 3.78604, 5.21941, 4.3393, 4.82896, 5.02483, 4.94016, 6.20228, 4.55212, 0.275813, 2.02912, 0.721371, 3.68215, 4.9445, 4.24451, 3.31991, 1.77942, 4.60286, 6.86935, 4.74788, 0.180684, 2.99859, 2.49172, 2.76458, 2.06447, 1.0738, 1.24119, 3.06678, 3.32671, 5.46126, 7.29366, 4.34522, 1.17638, 4.82303, 4.62953, 1.80883, 1.36018, 4.70439, 6.89125, 5.33115, 1.8484, 2.43964, 2.47957, 1.82924, 1.08836, 1.67501, 4.00404, 5.33944, 3.89576, 0.63757, 2.72817, 2.45126, 2.19431, 4.58439, 0.563696};

        EXCEPTION_ASSERT_EQUALS(expected.count (), sizeof(q)/sizeof(q[0]));

        double s=0;
        for (unsigned i=0; i<expected.count (); i++) {
            double d = p[i];
            d -= q[i];
            s = std::max(std::fabs(d), s);
        }

        // q is rounded to 5 decimals
        EXCEPTION_ASSERT_LESS(s, 0.5e-5);
    }

    // It should only accept StftDesc as TransformDesc.
    {
        EnvelopeDesc ed;

        EXPECT_EXCEPTION(ExceptionAssert,ed.transformDesc(Tfr::pTransformDesc()));
    }
}


} // namespace Filters
