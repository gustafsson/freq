#include "commentview.h"
#include "commentmodel.h"

#include <QLabel>

namespace Tools {

CommentView::
        CommentView(CommentModel* model)
            :
            enabled( false ),
            model_( model )
{
}


CommentView::
        ~CommentView()
{

}


void CommentView::
        draw()
{
    // TODO Use Qt Boxes demo
}


} // namespace Tools
