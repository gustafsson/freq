#ifndef TOOLS_SUPPORT_RENDEROPERATION_H
#define TOOLS_SUPPORT_RENDEROPERATION_H

#include "signal/operationwrapper.h"
#include "tfr/transform.h"

#include "volatileptr.h"

namespace Tools {
namespace Support {

/**
 * @brief The RenderOperationDesc class should keep the filter operation used by a
 * render target and notify that render target about processing events.
 */
class RenderOperationDesc : public Signal::OperationDescWrapper
{
public:
    class RenderTarget : public VolatilePtr<RenderTarget>
    {
    public:
        virtual ~RenderTarget() {}

        /**
         * @brief refreshSamples is called when samples are about to be recomputed.
         * @param I the samples that will be recomputed.
         */
        virtual void refreshSamples(const Signal::Intervals& I) = 0;

        /**
         * @brief processedData is called whenever new samples have been processed.
         */
        virtual void processedData() = 0;
    };


    RenderOperationDesc(Signal::OperationDesc::Ptr embed, RenderTarget::Ptr render_target);

    // Signal::OperationDescWrapper
    Signal::OperationWrapper*   createOperationWrapper(
                                    Signal::ComputingEngine*,
                                    Signal::Operation::Ptr wrapped) const;

    // Signal::OperationDesc
    Signal::Intervals           affectedInterval( const Signal::Intervals& I ) const;


    Tfr::TransformDesc::Ptr     transform_desc();
    void                        transform_desc(Tfr::TransformDesc::Ptr);

private:
    class Operation : public Signal::OperationWrapper
    {
    public:
        Operation(Signal::Operation::Ptr wrapped, RenderTarget::Ptr render_target);

        Signal::pBuffer process(Signal::pBuffer b);

    private:
        RenderTarget::Ptr render_target_;
    };

    RenderTarget::Ptr render_target_;

public:
    static void test();
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_RENDEROPERATION_H
