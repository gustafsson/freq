#ifndef TOOLSELECTOR_H
#define TOOLSELECTOR_H

class QWidget;

namespace Tools {
    class RenderView;

    namespace Support {

        class ToolSelector
        {
        public:
            ToolSelector(RenderView* render_view);

            QWidget* currentTool();
            void setCurrentTool(QWidget*);

        private:
            RenderView* _render_view;
            QWidget* _current_tool;
        };
    } // namespace Support
} // namespace Tools
#endif // TOOLSELECTOR_H
