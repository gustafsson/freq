#ifndef TOOLSELECTOR_H
#define TOOLSELECTOR_H

#include <QObject>
#include <QPointer>

class QWidget;

namespace Tools {
    class RenderView;

    namespace Support {

    /**
     Manages current tool

      A tool can only have one tool that receives all user actions
      such as QWidget::mouseMoveEvent, QWidget::wheelEvent, ...

      If a tool wants to extend another tool or have another tool as
      fallback for non implemented events it will have to create its own
      instance of the wanted 'default' tool and add itself as child.

      Preferably by adding the child tool using another ToolSelector.
     */
    class ToolSelector: public QObject
    {
    public:
        /// Sets what RenderView that is controlled by this ToolSelector.
        ToolSelector(QWidget* parent_tool);


        /// @see setCurrentTool
        QWidget* currentTool();

        /// @see ToolSelector(QWidget*)
        QWidget* parentTool();

        QPointer<QWidget> default_tool;
        QPointer<QWidget> temp_tool;


        /**
          Makes tool a child of _render_view, disables any previous tool.

          The tool can find out when it is enabled and disabled by implementing
          the changeEvent() handler and check if type is QEvent::EnabledChange.

          If a tool doesn't wan't to render any Qt widgets it should set its
          attributes accordingly with:
            'setAttribute(Qt::WA_DontShowOnScreen, true)'
          To draw stuff in the 3d scene atop the heightmap the tool should
          connect to the appropriate signals in RenderView.
          */
        void setCurrentTool(QWidget* tool, bool active );

    private:
        QWidget* _parent_tool;
        QWidget* _current_tool;
        bool _must_have_one_tool;
    };
}
} // namespace Tools
#endif // TOOLSELECTOR_H
