#include "recordview.h"

#include "recordmodel.h"

namespace Tools
{

RecordView::
        RecordView(RecordModel* model)
            :
            enabled(false),
            model_(model)
{}


RecordView::
        ~RecordView()
{

}


void RecordView::
        draw()
{

}


} // namespace Tools
