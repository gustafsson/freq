#include "renderoperation.h"
#include "tfr/chunkfilter.h"
#include "tfr/transformoperation.h"

using namespace Signal;

namespace Tools {
namespace Support {


RenderOperationDesc::
        RenderOperationDesc(OperationDesc::ptr embed, RenderTarget::ptr render_target)
    :
      OperationDescWrapper(embed),
      render_target_(render_target)
{
}


Operation::ptr RenderOperationDesc::
        createOperation(ComputingEngine* engine=0) const
{
    Operation::ptr wrap = OperationDescWrapper::createOperation (engine);

    if (!wrap)
        return Operation::ptr();

    return Operation::ptr(new Operation(wrap, render_target_));
}


Interval RenderOperationDesc::
        affectedInterval( const Interval& I ) const
{
    const Interval& a = OperationDescWrapper::affectedInterval( I );

    // This will result in an update rate that matches the invalidated intervals if possible.
    render_target_.write ()->refreshSamples( a );

    return a;
}


Tfr::TransformDesc::ptr RenderOperationDesc::
        transform_desc() const
{
    Signal::OperationDesc::ptr wo = getWrappedOperationDesc();
    if (!wo)
        return Tfr::TransformDesc::ptr();

    auto o = wo.read ();
    const Tfr::TransformOperationDesc* f = dynamic_cast<const Tfr::TransformOperationDesc*>(&*o);
    if (f)
        return f->transformDesc ();

    return Tfr::TransformDesc::ptr();
}


void RenderOperationDesc::
        transform_desc(Tfr::TransformDesc::ptr t)
{
    Signal::OperationDesc::ptr wo = getWrappedOperationDesc();
    if (!wo)
        return;

    auto o = wo.write ();
    Tfr::TransformOperationDesc* f = dynamic_cast<Tfr::TransformOperationDesc*>(&*o);
    if (f)
        f->transformDesc (t);
}


RenderOperationDesc::Operation::
        Operation(Operation::ptr wrapped, RenderTarget::ptr render_target)
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

    b = wrapped_.write ()->process (b);

    Signal::Interval output = b?b->getInterval ():Signal::Interval();

    render_target_.write ()->processedData (input, output);

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

        OperationDesc::ptr operation( new Test::TransparentOperationDesc() );
        RenderTarget::ptr rtp(target = new RenderOperationDescMockTarget());

        Signal::OperationDesc::ptr ro(rod = new RenderOperationDesc(operation, rtp));
        Signal::Operation::ptr o = ro.write ()->createOperation(0);

        // Operations are processed through a Processing::Step
        Processing::Step step(ro);
        step.deprecateCache (Interval(4,9));
        o.write ()->process (pBuffer());

        EXCEPTION_ASSERT_EQUALS( Interval(4,9), target->I );
        EXCEPTION_ASSERT_EQUALS( 1, target->processed_count );
        EXCEPTION_ASSERT_EQUALS( (void*)0, rod->transform_desc ().get () );
    }
}

} // namespace Support
} // namespace Tools
