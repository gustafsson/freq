#ifndef TOOLS_SUPPORT_RENDEROPERATION_H
#define TOOLS_SUPPORT_RENDEROPERATION_H

#include "signal/operationwrapper.h"
#include "tools/renderview.h"

namespace Tools {
namespace Support {

class RenderOperation : public Signal::OperationWrapper
{
public:
    RenderOperation(Signal::Operation::Ptr wrapped, RenderView* view);

    Signal::pBuffer process(Signal::pBuffer b);

private:
    RenderView* view_;
};


class RenderOperationDesc : public Signal::OperationDescWrapper
{
public:
    RenderOperationDesc(Signal::OperationDesc::Ptr embed, RenderView* view);

    // Signal::OperationDescWrapper
    Signal::OperationWrapper*   createOperationWrapper(
                                    Signal::ComputingEngine*,
                                    Signal::Operation::Ptr wrapped) const;

    // Signal::OperationDesc
    Signal::Intervals           affectedInterval( const Signal::Intervals& I ) const;


    Tfr::TransformDesc::Ptr     transform_desc();
    void                        transform_desc(Tfr::TransformDesc::Ptr);

private:
    RenderView* view_;
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_RENDEROPERATION_H
