#include "toolselector.h"

#include "tools/renderview.h"
#include "sawe/project.h"
#include "ui/mainwindow.h"

// Qt
#include <QHBoxLayout>
#include <QWidget>

namespace Tools {
    namespace Support {


ToolSelector::
        ToolSelector(RenderView* render_view)
            :
            _render_view(render_view)
{

}


QWidget* ToolSelector::
        currentTool()
{
    return _current_tool;
}


void ToolSelector::
        setCurrentTool( QWidget* tool )
{
    // If the tool doesn't have a layout policy, create one for it. QHBoxLayout
    // makes children (i.e _render_view) expand to fill the entire tool widget.
    // A tool implementation can break layout if it wants to. This code works
    // with widgets created with default settings as in "new QWidget()".
    if (!tool->layout())
    {
        tool->setLayout( new QHBoxLayout );
        tool->layout()->setMargin(0);
    }

    // Put render view as a child of tool, this effectively removes render
    // view as child of any previous tool.
    // Since _render_view doesn't implement any event handlers all events on
    // the render view are passed on to tool.
    tool->layout()->addWidget( _render_view );

    if (_current_tool)
    {
        // Remove the current tool from the central widget. Memory management
        // is supposed to be taken care of by someone else
        _current_tool->setParent( 0 );
        _current_tool->setEnabled( false );
    }

    // Add the tool as a child (the only child) of the central widget
    _render_view->model->project()->mainWindow()->centralWidget()->layout()->addWidget( tool );
    _current_tool = tool;
    _current_tool->setEnabled( true );

    // The tool can find out when it is enabled and disabled by implementing the
    // changeEvent() handler with type QEvent::EnabledChange.
}

    } // namespace Support
} // namespace Tools
