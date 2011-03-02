#include "toolbar.h"

namespace Tools { namespace Support {

ToolBar::ToolBar(QWidget *parent) :
    QToolBar(parent)
{
}


void ToolBar::
        showEvent(QShowEvent*)
{
    emit visibleChanged( true );
}


void ToolBar::
        hideEvent(QHideEvent*)
{
    emit visibleChanged( false );
}

} // namespace Support
} // namespace Tools
