#ifndef TOOLFACTORY_H
#define TOOLFACTORY_H

namespace Sawe {
    class Project;
}

// TODO remove
#include "rendermodel.h"
#include "selectionmodel.h"
#include "playbackmodel.h"

#include <typeinfo>

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

        RenderModel render_model;
        SelectionModel selection_model;
        PlaybackModel playback_model;

    private:
        class RenderController* _render_controller;
        class RenderView* _render_view;

        class TimelineView* _timeline_view;
        class TimelineController* _timeline_controller;

        class SelectionView* _selection_view;
        class SelectionController* _selection_controller;

        Sawe::Project* _project;
    };
} // namespace Tools

#endif // TOOLFACTORY_H
