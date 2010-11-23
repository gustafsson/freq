#include "commentview2.h"
#include "commentmodel.h"
#include "ui/mousecontrol.h"
#include "renderview.h"
#include <QLabel>

namespace Tools {

CommentView2::
        CommentView2(RenderView* render_view)
            :
            qx(0), qy(0), qz(0),
            render_view_( render_view )
{
}


CommentView2::
        ~CommentView2()
{
}


void CommentView2::
        updatePosition()
{
}

} // namespace Tools
