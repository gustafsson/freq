#include "selectionmodel.h"
#include "sawe/project.h"

#include "filters/selection.h"
#include "tfr/transformoperation.h"

#include "tasktimer.h"
#include "demangle.h"

namespace Tools
{

SelectionModel::
        SelectionModel(Sawe::Project* p)
            : project_(p)
{
}


SelectionModel::
        ~SelectionModel()
{
    TaskInfo(__FUNCTION__);
}


void SelectionModel::
        set_current_selection(Signal::OperationDesc::Ptr o)
{
    if (o!=current_selection_) {
        // Check if 'o' is supported by making a copy of it
        current_selection_ = copy_selection( o );
    }

    emit selectionChanged();
}


void SelectionModel::
        try_set_current_selection(Signal::OperationDesc::Ptr o)
{
    TaskInfo ti("Trying to set %s \"%s\" as current selection", vartype(*o.get()).c_str(), o.read ()->toString().toStdString().c_str());

    try
    {
        set_current_selection( o );
    }
    catch ( const std::logic_error& )
    {
        set_current_selection(Signal::OperationDesc::Ptr());
    }
}


Signal::OperationDesc::Ptr SelectionModel::
        current_selection_copy(SaveInside si)
{
    return copy_selection( current_selection_, si );
}


Signal::OperationDesc::Ptr SelectionModel::
        copy_selection(Signal::OperationDesc::Ptr o, SaveInside si)
{
    if (!o)
        return o;

    o = o.read ()->copy();
    if (si == SaveInside_UNCHANGED)
        return o;

    auto w = o.write ();

    if (Filters::Selection* s = dynamic_cast<Filters::Selection*>( &*w ))
      {
        s->selectInterior (si == SaveInside_TRUE);
        return o;
      }

    if (Tfr::TransformOperationDesc* t = dynamic_cast<Tfr::TransformOperationDesc*>( &*w ))
      {
        auto cfd = t->chunk_filter();
        if (Filters::Selection* s = dynamic_cast<Filters::Selection*>( &*cfd ))
          {
            s->selectInterior (si == SaveInside_TRUE);
            return o;
          }
      }

    EXCEPTION_ASSERTX(false, "SelectionModel::copy_selection(" + vartype(*o.raw ()) + ", " + w->toString().toStdString() + ") is not implemented");
    return Signal::OperationDesc::Ptr();
}



} // namespace Tools
