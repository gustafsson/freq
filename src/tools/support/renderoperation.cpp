#include "renderoperation.h"
#include "signal/oldoperationwrapper.h"
#include "tfr/filter.h"

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


OperationWrapper* RenderOperationDesc::
        createOperationWrapper(ComputingEngine*, Operation::Ptr wrapped) const
{
    return new Operation(wrapped, render_target_);
}


Intervals RenderOperationDesc::
        affectedInterval( const Intervals& I ) const
{
    const Intervals& a = OperationDescWrapper::affectedInterval( I );

    // This will result in a update rate that matches the invalidated intervals if possible.
    write1(render_target_)->refreshSamples( a );

    return a;
}


Tfr::TransformDesc::Ptr RenderOperationDesc::
        transform_desc()
{
    // TODO use Processing
    OperationDesc::Ptr o = getWrappedOperationDesc();
    Tfr::FilterDesc* f = dynamic_cast<Tfr::FilterDesc*>(o.get ());
    if (f)
        return f->transformDesc ();

    OldOperationDescWrapper* w = dynamic_cast<OldOperationDescWrapper*>(o.get ());
    if (w)
    {
        Tfr::Filter* f2 = dynamic_cast<Tfr::Filter*>(w->old_operation ().get ());
        if (f2)
            return f2->transform ()->transformDesc ()->copy ();
    }
    return Tfr::TransformDesc::Ptr();
}


void RenderOperationDesc::
        transform_desc(Tfr::TransformDesc::Ptr t)
{
    // TODO use Processing
    OperationDesc::Ptr o = getWrappedOperationDesc();
    Tfr::FilterDesc* f = dynamic_cast<Tfr::FilterDesc*>(o.get ());
    if (f)
        return f->transformDesc (t);

    OldOperationDescWrapper* w = dynamic_cast<OldOperationDescWrapper*>(o.get ());
    if (w)
    {
        Tfr::Filter* f2 = dynamic_cast<Tfr::Filter*>(w->old_operation ().get ());
        if (f2)
            return f2->transform (t->createTransform ());
    }
}


RenderOperationDesc::Operation::
        Operation(Operation::Ptr wrapped, RenderTarget::Ptr render_target)
    :
      OperationWrapper(wrapped),
      render_target_(render_target)
{
    EXCEPTION_ASSERT( wrapped );
    EXCEPTION_ASSERT( render_target );
}


pBuffer RenderOperationDesc::Operation::
        process(pBuffer b)
{
    Signal::Interval input = b?b->getInterval ():Signal::Interval();

    b = OperationWrapper::process (b);

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
        step.operation(ComputingEngine::Ptr())->process (pBuffer());

        EXCEPTION_ASSERT_EQUALS( Interval(4,9), target->I );
        EXCEPTION_ASSERT_EQUALS( 1, target->processed_count );
        EXCEPTION_ASSERT_EQUALS( (void*)0, rod->transform_desc ().get () );
    }
}

} // namespace Support
} // namespace Tools
