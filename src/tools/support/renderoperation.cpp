#include "renderoperation.h"
#include "tfr/chunkfilter.h"
#include "tfr/transformoperation.h"

using namespace Signal;

namespace Tools {
namespace Support {


RenderOperationDesc::
        RenderOperationDesc(OperationDesc::Ptr embed, RenderTarget::Ptr render_target)
    :
      OperationDescWrapper(embed),
      render_target_(render_target)
{
}


Operation::Ptr RenderOperationDesc::
        createOperation(ComputingEngine* engine=0) const
{
    Operation::Ptr wrap = OperationDescWrapper::createOperation (engine);

    if (!wrap)
        return Operation::Ptr();

    return Operation::Ptr(new Operation(wrap, render_target_));
}


Interval RenderOperationDesc::
        affectedInterval( const Interval& I ) const
{
    const Interval& a = OperationDescWrapper::affectedInterval( I );

    // This will result in an update rate that matches the invalidated intervals if possible.
    write1(render_target_)->refreshSamples( a );

    return a;
}


Tfr::TransformDesc::Ptr RenderOperationDesc::
        transform_desc() const
{
    Signal::OperationDesc::Ptr wo = getWrappedOperationDesc();
    if (!wo)
        return Tfr::TransformDesc::Ptr();

    OperationDesc::ReadPtr o(wo);
    const Tfr::TransformOperationDesc* f = dynamic_cast<const Tfr::TransformOperationDesc*>(&*o);
    if (f) {
        Tfr::ChunkFilterDesc::Ptr c = f->chunk_filter ();
        return write1(c)->transformDesc ();
    }

    return Tfr::TransformDesc::Ptr();
}


void RenderOperationDesc::
        transform_desc(Tfr::TransformDesc::Ptr t)
{
    Signal::OperationDesc::Ptr wo = getWrappedOperationDesc();
    if (!wo)
        return;

    OperationDesc::ReadPtr o(wo);
    const Tfr::TransformOperationDesc* f = dynamic_cast<const Tfr::TransformOperationDesc*>(&*o);
    if (f) {
        Tfr::ChunkFilterDesc::Ptr c = f->chunk_filter ();
        write1(c)->transformDesc (t);
    }
}


RenderOperationDesc::Operation::
        Operation(Operation::Ptr wrapped, RenderTarget::Ptr render_target)
    :
      wrapped_(wrapped),
      render_target_(render_target)
{
    EXCEPTION_ASSERT( wrapped );
    EXCEPTION_ASSERT( render_target );
}


pBuffer RenderOperationDesc::Operation::
        process(pBuffer b)
{
    Signal::Interval input = b?b->getInterval ():Signal::Interval();

    b = write1(wrapped_)->process (b);

    Signal::Interval output = b?b->getInterval ():Signal::Interval();

    write1(render_target_)->processedData (input, output);

    return b;
}

} // namespace Support
} // namespace Tools

#include "test/operationmockups.h"
#include "signal/processing/step.h"

namespace Tools {
namespace Support {

class RenderOperationDescMockTarget: public RenderOperationDesc::RenderTarget
{
public:
    RenderOperationDescMockTarget()
        :
          processed_count(0)
    {}


    void refreshSamples(const Intervals& I) {
        this->I = I;
    }


    void processedData(const Interval&, const Interval&) {
        processed_count++;
    }

    Intervals I;
    int processed_count;
};


void RenderOperationDesc::
    test()
{
    // The RenderOperationDesc class should keep the filter operation used by a
    // render target and notify the render target about processing events.
    {
        RenderOperationDesc* rod;
        RenderOperationDescMockTarget* target;

        OperationDesc::Ptr operation( new Test::TransparentOperationDesc() );
        RenderTarget::Ptr rtp(target = new RenderOperationDescMockTarget());

        Signal::OperationDesc::Ptr ro(rod = new RenderOperationDesc(operation, rtp));

        // Operations are processed through a Processing::Step
        Processing::Step step(ro);
        step.deprecateCache (Interval(4,9));
        write1(step.operation(ComputingEngine::Ptr()))->process (pBuffer());

        EXCEPTION_ASSERT_EQUALS( Interval(4,9), target->I );
        EXCEPTION_ASSERT_EQUALS( 1, target->processed_count );
        EXCEPTION_ASSERT_EQUALS( (void*)0, rod->transform_desc ().get () );
    }
}

} // namespace Support
} // namespace Tools
