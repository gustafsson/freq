#ifndef SELECTIONVIEW_H
#define SELECTIONVIEW_H

#include <QObject>

namespace Tools
{
    class SelectionModel;

    /**
      Models shall always live longer than their corresponding views.
      */
    class SelectionView: public QObject
    {
        Q_OBJECT
    public:
        SelectionView(SelectionModel* model);
        ~SelectionView();

        void drawSelection();
        void drawSelectionSquare();
        bool insideCircle( float x1, float z1 );
        void drawSelectionCircle();
        void drawSelectionCircle2();

        bool enabled;

    public slots:
        /// Connected in SelectionController
        virtual void draw();

    private:
        friend class SelectionController;
        SelectionModel* model;
    };
} // namespace Tools

#endif // SELECTIONVIEW_H
