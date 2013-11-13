#include "toolselector.h"

#include "tools/commands/changetoolcommand.h"

#include "sawe/project.h"
#include "ui/mainwindow.h"
#include "demangle.h"
// Qt
#include <QHBoxLayout>
#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsProxyWidget>

namespace Tools {
    namespace Support {


ToolSelector::
        ToolSelector(Tools::Commands::CommandInvoker* command_invoker, QWidget* parent_tool)
            :
            _command_invoker(command_invoker),
            _parent_tool(parent_tool),
            _current_tool(0),
            _must_have_one_tool(true)
{

}


QWidget* ToolSelector::
        currentTool()
{
    return _current_tool;
}


QWidget* ToolSelector::
        parentTool()
{
    return _parent_tool;
}


//class CustomGraphicsProxy: public QGraphicsProxyWidget
//{
//public:
//    CustomGraphicsProxy() {}

//    void paint(QPainter *, const QStyleOptionGraphicsItem *, QWidget *)
//    {}
//};

// "inline" suppresses warning about unused function
inline static void printChildren(QObject* o)
{
    TaskTimer tt("%s", vartype(*o).c_str());
    tt.suppressTiming();

    const QObjectList& c = o->children();
    foreach( QObject*i, c )
    {
        printChildren(i);
    }
}


void ToolSelector::
        setCurrentTool( QWidget* tool, bool active )
{
    if ((tool != _current_tool) != active)
        return;

    if (!active)
    {
        if (default_tool && default_tool != tool)
        {
            tool = default_tool;
        }
        else if (_must_have_one_tool)
        {
            // ignore disable
            TaskInfo("Can't disable tool %s in %s because it must have one tool",
                     vartype(*tool).c_str(),
                     vartype(*_parent_tool).c_str());
            return;
        }
        else
        {
            // new current tool will be 0
            tool = 0;
        }
    }

    if (tool!=_current_tool)
    {
        Tools::Commands::pCommand cmd( new Tools::Commands::ChangeToolCommand(tool, this));
        this->_command_invoker->invokeCommand( cmd );
    }
}

void ToolSelector::
        setCurrentToolCommand( QWidget* tool )
{
    if (_current_tool && _current_tool != tool)
    {
        TaskInfo("Current tool in %s was %s",
            vartype(*_parent_tool).c_str(),
            vartype(*_current_tool).c_str());

        // Remove the current tool from the render view. Memory management
        // is supposed to be taken care of by someone else. QPointer is a
        // good way of handling memory managment of QObject, as is done in
        // toolfactory.
        QPointer<QWidget> prev_current_tool = _current_tool;
        _current_tool = 0;
        prev_current_tool->setEnabled( false );

        // _current_tool->setEnabled might change delete prev_current_tool
        if (prev_current_tool)
        {
            prev_current_tool->setParent( 0 );
            _parent_tool->update();
        }
    }

    _current_tool = tool;

    if (_current_tool)
    {
        // Put tool as a child of _render_view.
        _current_tool->setParent(_parent_tool);
        _parent_tool->layout()->addWidget( _current_tool );

        _current_tool->setEnabled( true );
        _parent_tool->update();
    }

    TaskInfo("Current tool in %s is %s",
        vartype(*_parent_tool).c_str(),
        _current_tool?vartype(*_current_tool).c_str():0);
}


    } // namespace Support
} // namespace Tools
