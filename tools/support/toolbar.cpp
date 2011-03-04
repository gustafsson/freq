#include "toolbar.h"

namespace Tools { namespace Support {

ToolBar::ToolBar(QWidget *parent) :
    QToolBar(parent)
{
}


void ToolBar::
        showEvent(QShowEvent*)
{
    emit visibleChanged( isVisible() );
}


void ToolBar::
        hideEvent(QHideEvent*)
{
    emit visibleChanged( isVisible() );
}

} // namespace Support
} // namespace Tools
