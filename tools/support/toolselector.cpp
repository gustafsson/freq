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
            _render_view(render_view),
            _current_tool(0)
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
    if (tool != _current_tool)
    {
        if (_current_tool)
        {
            // Remove the current tool from the render view. Memory management
            // is supposed to be taken care of by someone else. QPointer is a
            // good way of handling memory managment of QObject, as is done in
            // toolfactory.
            _current_tool->setParent( 0 );
            _current_tool->setEnabled( false );
        }

        _current_tool = tool;

        if (_current_tool)
        {
            // Put tool as a child of _render_view.
            _render_view->layout()->addWidget( _current_tool );
            _current_tool->setEnabled( true );
        }
    }
}

    } // namespace Support
} // namespace Tools
