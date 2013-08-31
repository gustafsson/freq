#include "renderoperation.h"
#include "signal/oldoperationwrapper.h"
#include "tfr/filter.h"

namespace Tools {
namespace Support {

RenderOperation::
        RenderOperation(Signal::Operation::Ptr wrapped, RenderView* view)
    :
      Signal::OperationWrapper(wrapped),
      view_(view)
{
    EXCEPTION_ASSERT( wrapped );
    EXCEPTION_ASSERT( view );
}


Signal::pBuffer RenderOperation::
        process(Signal::pBuffer b)
{
    b = Signal::OperationWrapper::process (b);

    view_->userinput_update ();

    return b;
}


RenderOperationDesc::
        RenderOperationDesc(Signal::OperationDesc::Ptr embed, RenderView* view)
    :
      OperationDescWrapper(embed),
      view_(view)
{
}


Signal::OperationWrapper* RenderOperationDesc::
        createOperationWrapper(Signal::ComputingEngine*, Signal::Operation::Ptr wrapped) const
{
    return new RenderOperation(wrapped, view_);
}


Signal::Intervals RenderOperationDesc::
        affectedInterval( const Signal::Intervals& I ) const
{
    const Signal::Intervals& a = Signal::OperationDescWrapper::affectedInterval( I );

    // This will result in a update rate that matches the invalidated intervals if possible.
    view_->setLastUpdateSize( a.count () );

    return a;
}


Tfr::TransformDesc::Ptr RenderOperationDesc::
        transform_desc()
{
    // TODO use Signal::Processing
    Signal::OperationDesc::Ptr o = getWrappedOperationDesc();
    Tfr::FilterDesc* f = dynamic_cast<Tfr::FilterDesc*>(o.get ());
    if (f)
        return f->transformDesc ();

    Signal::OldOperationDescWrapper* w = dynamic_cast<Signal::OldOperationDescWrapper*>(o.get ());
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
    // TODO use Signal::Processing
    Signal::OperationDesc::Ptr o = getWrappedOperationDesc();
    Tfr::FilterDesc* f = dynamic_cast<Tfr::FilterDesc*>(o.get ());
    if (f)
        return f->transformDesc (t);

    Signal::OldOperationDescWrapper* w = dynamic_cast<Signal::OldOperationDescWrapper*>(o.get ());
    if (w)
    {
        Tfr::Filter* f2 = dynamic_cast<Tfr::Filter*>(w->old_operation ().get ());
        if (f2)
            return f2->transform (t->createTransform ());
    }
}

} // namespace Support
} // namespace Tools
