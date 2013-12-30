#ifndef TOOLS_SUPPORT_DRAWWAVEFORM_H
#define TOOLS_SUPPORT_DRAWWAVEFORM_H

#include "signal/operation.h"

namespace Tools {
    namespace Support {

class DrawWaveform
{
public:
    DrawWaveform();

    void drawWaveform_chunk_directMode( Signal::pBuffer chunk);

private:
    struct ListCounter {
        unsigned displayList;
        enum Age {
            Age_JustCreated,
            Age_InUse,
            Age_ProposedForRemoval
        } age;
        //ListAge age;
    };
    std::map<void*, ListCounter> _chunkGlList;
    bool _enqueueGcDisplayList;

    template<typename RenderData>
    void draw_glList(
            boost::shared_ptr<RenderData> chunk,
            void (*renderFunction)( boost::shared_ptr<RenderData> ),
            bool force_redraw );
    void gcDisplayList();
};

    } // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_DRAWWAVEFORM_H
