#include "commentview.h"
#include "commentmodel.h"
#include "ui/mousecontrol.h"
#include "renderview.h"
#include <QLabel>

namespace Tools {

CommentView::
        CommentView(RenderView* render_view)
            :
            qx(0), qy(0), qz(0),
            render_view_( render_view )
{
}


CommentView::
        ~CommentView()
{
}


void CommentView::
        updatePosition()
{
}

} // namespace Tools
