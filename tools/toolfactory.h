#ifndef TOOLFACTORY_H
#define TOOLFACTORY_H

namespace Sawe {
    class Project;
}

// TODO remove
#include "rendermodel.h"
#include "selectionmodel.h"

#include "selectionview.h"

namespace Tools
{
    /**
      Find a better name...
      */
    class ToolFactory
    {
    public:
        ToolFactory(Sawe::Project* p);
        ~ToolFactory();

        // TODO remove
        RenderModel render_model;
        SelectionModel selection_model;

        SelectionView selection_view;

        // PlaybackView playback_view;
        // DiskwriterView diskwriter_view;

        // map<string, QWidget*> SelectionView selection_widget;

    private:
        class RenderController* _render_controller;
        class RenderView* _render_view;

        class TimelineView* _timeline_view;
        class TimelineController* _timeline_controller;

        Sawe::Project* _project;
    };
} // namespace Tools

#endif // TOOLFACTORY_H
