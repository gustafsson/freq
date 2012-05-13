#include "toolbar.h"

namespace Tools { namespace Support {

ToolBar::ToolBar(QWidget *parent) :
    QToolBar(parent)
{
}


void ToolBar::
        showEvent(QShowEvent*)
{
    emit visibleChanged( isVisibleTo( parentWidget() ) );
}


void ToolBar::
        hideEvent(QHideEvent*)
{
    emit visibleChanged( isVisibleTo( parentWidget() ) );
}

} // namespace Support
} // namespace Tools
