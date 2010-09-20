#include "selectionview.h"

#include <QWidget>

namespace Tools
{
    class SelectionController: public QWidget
    {
        Q_OBJECT
    public:
        SelectionController( QWidget* parent, SelectionView* view);

    private:
        SelectionView* view;
    };
}
