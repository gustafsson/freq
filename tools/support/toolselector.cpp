#include "toolselector.h"

#include "sawe/project.h"
#include "ui/mainwindow.h"

// Qt
#include <QHBoxLayout>
#include <QWidget>

namespace Tools {
    namespace Support {


ToolSelector::
        ToolSelector(QWidget* parent_tool)
            :
            _parent_tool(parent_tool),
            _current_tool(0)
{

}


QWidget* ToolSelector::
        currentTool()
{
    return _current_tool;
}


void ToolSelector::
        setCurrentTool( QWidget* tool, bool active )
{
    if ((tool != _current_tool) == active)
    {
        if (_current_tool)
        {
            // Remove the current tool from the render view. Memory management
            // is supposed to be taken care of by someone else. QPointer is a
            // good way of handling memory managment of QObject, as is done in
            // toolfactory.
            _current_tool->setParent( 0 );
            _current_tool->setEnabled( false );
            _current_tool = 0;
            _parent_tool->update();
        }
    }

    if (active)
    {
        _current_tool = tool;

        if (_current_tool)
        {
            // Put tool as a child of _render_view.
            _parent_tool->layout()->addWidget( _current_tool );
            _current_tool->setEnabled( true );
            _parent_tool->update();
        }
    }
}


    } // namespace Support
} // namespace Tools
