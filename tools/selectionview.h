#ifndef SELECTIONVIEW_H
#define SELECTIONVIEW_H

#include "selectionmodel.h"

namespace Tools
{
    /**
      Models shall always live longer than their corresponding views.
      */
    class SelectionView
    {
    public:
        SelectionView(SelectionModel* model);

        void drawSelection();
        void drawSelectionSquare();
        bool insideCircle( float x1, float z1 );
        void drawSelectionCircle();
        void drawSelectionCircle2();
        void drawPlaybackMarker();

        float _playbackMarker;
    private:
        SelectionModel* model;
    };
} // namespace Tools

#endif // SELECTIONVIEW_H
