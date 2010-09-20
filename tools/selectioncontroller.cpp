#include "selectioncontroller.h"

namespace Tools
{
    SelectionController::SelectionController( QWidget* parent, SelectionView* view)
        :   QWidget(parent),
            view(view)
    {
    }

} // namespace Tools
