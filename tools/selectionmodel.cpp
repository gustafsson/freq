#include "selectionmodel.h"
#include "sawe/project.h"

#include "signal/operation-basic.h"
#include "support/operation-composite.h"
#include "filters/ellipse.h"
#include "filters/rectangle.h"
#include "filters/bandpass.h"
#include "tools/selections/support/splinefilter.h"

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
        set_current_selection(Signal::pOperation o)
{
    if (o==current_selection_)
        return;

    // Check if 'o' is supported by making a copy of it
    current_selection_ = copy_selection( o );

    emit selectionChanged();
}


void SelectionModel::
        try_set_current_selection(Signal::pOperation o)
{
    TaskInfo ti("Trying to set %s \"%s\" as current selection", vartype(*o.get()).c_str(), o->name().c_str());

    try
    {
        set_current_selection( o );
    }
    catch ( const std::logic_error& )
    {
        set_current_selection(Signal::pOperation());
    }
}


Signal::pOperation SelectionModel::
        current_selection_copy(SaveInside si)
{
    return copy_selection( current_selection_, si );
}


template<>
Signal::pOperation SelectionModel::
        copy_selection_type(Filters::Ellipse* src, SaveInside si)
{
    Filters::Ellipse* dst;
    Signal::pOperation o( dst=new Filters::Ellipse( *src ));

    if (si != SaveInside_UNCHANGED)
        dst->_save_inside = si == SaveInside_TRUE;
    return o;
}


template<>
Signal::pOperation SelectionModel::
        copy_selection_type(Filters::Rectangle* src, SaveInside si)
{
    Filters::Rectangle* dst;
    Signal::pOperation o( dst=new Filters::Rectangle( *src ));

    if (si != SaveInside_UNCHANGED)
        dst->_save_inside = si == SaveInside_TRUE;
    return o;
}


template<>
Signal::pOperation SelectionModel::
        copy_selection_type(Filters::Bandpass* src, SaveInside si)
{
    Filters::Bandpass* dst;
    Signal::pOperation o( dst=new Filters::Bandpass( *src ));

    if (si != SaveInside_UNCHANGED)
        dst->_save_inside = si == SaveInside_TRUE;
    return o;
}


template<>
Signal::pOperation SelectionModel::
        copy_selection_type(Tools::Support::OperationOtherSilent* src, SaveInside si)
{
    if (si == SaveInside_UNCHANGED || si == SaveInside_TRUE) {
        return Signal::pOperation( new Tools::Support::OperationOtherSilent( *src ));
    } else {
        return Signal::pOperation( new Signal::OperationSetSilent(
                Signal::pOperation(),
                src->section() )
        );
    }
}


template<>
Signal::pOperation SelectionModel::
        copy_selection_type(Signal::OperationSetSilent* src, SaveInside si)
{
    if (si == SaveInside_UNCHANGED || si == SaveInside_TRUE) {
        return Signal::pOperation( new Signal::OperationSetSilent( *src ));
    } else {
        return Signal::pOperation( new Tools::Support::OperationOtherSilent(
                Signal::pOperation(),
                src->affected_samples().coveredInterval() )
        );
    }
}


template<>
Signal::pOperation SelectionModel::
        copy_selection_type(Selections::Support::SplineFilter* src, SaveInside si)
{
    Selections::Support::SplineFilter* dst;
    Signal::pOperation o( dst=new Selections::Support::SplineFilter( *src ));

    if (si != SaveInside_UNCHANGED)
        dst->_save_inside = si == SaveInside_TRUE;
    return o;
}


Signal::pOperation SelectionModel::
        copy_selection(Signal::pOperation o, SaveInside si)
{
    if (!o)
        return o;

#define TEST_TYPE(T) \
    do { \
        T* p = dynamic_cast<T*>( o.get() ); \
        if (p) return copy_selection_type(p, si); \
    } while(false)

    TEST_TYPE(Filters::Ellipse);
    TEST_TYPE(Filters::Rectangle);
    TEST_TYPE(Filters::Bandpass);
    TEST_TYPE(Tools::Support::OperationOtherSilent);
    TEST_TYPE(Signal::OperationSetSilent);
    TEST_TYPE(Selections::Support::SplineFilter);

    throw std::logic_error("SelectionModel::copy_selection(" + vartype(*o) + ") is not implemented");
}



} // namespace Tools
