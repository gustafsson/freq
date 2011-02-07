#include "toolselector.h"

#include "sawe/project.h"
#include "ui/mainwindow.h"
#include <demangle.h>
// Qt
#include <QHBoxLayout>
#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsProxyWidget>

namespace Tools {
    namespace Support {


ToolSelector::
        ToolSelector(QWidget* parent_tool)
            :
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

static void printChildren(QObject* o)
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
    if ((tool != _current_tool) == active)
    {
        if (active || !_must_have_one_tool) if (_current_tool)
        {
            TaskInfo("Current tool in %s was %s",
                vartype(*_parent_tool).c_str(),
                vartype(*_current_tool).c_str());

            // Remove the current tool from the render view. Memory management
            // is supposed to be taken care of by someone else. QPointer is a
            // good way of handling memory managment of QObject, as is done in
            // toolfactory.
            _current_tool->setEnabled( false );
            if (_current_tool)
            {
                _current_tool->setParent( 0 );
                _current_tool = 0;
                _parent_tool->update();
            }
        }
    }

    if (active)
    {
        _current_tool = tool;

        if (_current_tool)
        {
            // Put tool as a child of _render_view.
            _current_tool->setParent(_parent_tool);
            _parent_tool->layout()->addWidget( _current_tool );
/*
            CustomGraphicsProxy* proxy = new CustomGraphicsProxy();
            _current_tool->move(-INT_MAX/2, -INT_MAX/2);
            _current_tool->resize( QSize(INT_MAX, INT_MAX) );
            proxy->setScale( 2 );
            proxy->setWidget( _current_tool );
            proxy->setAttribute( Qt::WA_DontShowOnScreen, true );
            BOOST_ASSERT( dynamic_cast<QGraphicsView*>(_parent_tool) );
            dynamic_cast<QGraphicsView*>(_parent_tool)->scene()->addItem( proxy );

            QObject* p = _parent_tool;
            while (p->parent()) p = p->parent();
            printChildren(p);*/

            _current_tool->setEnabled( true );
            _parent_tool->update();
        }
    }
    TaskInfo("Current tool in %s is %s",
        vartype(*_parent_tool).c_str(),
        vartype(*_current_tool).c_str());
}


    } // namespace Support
} // namespace Tools
