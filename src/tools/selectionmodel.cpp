#include "selectionmodel.h"
#include "sawe/project.h"

#include "signal/operation-basic.h"
#include "support/operation-composite.h"
#include "filters/ellipse.h"
#include "filters/rectangle.h"
#include "filters/bandpass.h"
//#include "tools/selections/support/splinefilter.h"

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
    TaskInfo ti("Trying to set %s \"%s\" as current selection", vartype(*o.get()).c_str(), read1(o)->toString().toStdString().c_str());

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


template<>
Signal::OperationDesc::Ptr SelectionModel::
        set_parity(Signal::OperationDesc::Ptr o, Filters::Ellipse* src, SaveInside si)
{
    if (si != SaveInside_UNCHANGED)
        src->_save_inside = si == SaveInside_TRUE;
    src->updateChunkFilter ();
    return o;
}


template<>
Signal::OperationDesc::Ptr SelectionModel::
        set_parity(Signal::OperationDesc::Ptr o, Filters::Rectangle* src, SaveInside si)
{
    if (si != SaveInside_UNCHANGED)
        src->_save_inside = si == SaveInside_TRUE;
    src->updateChunkFilter ();
    return o;
}


template<>
Signal::OperationDesc::Ptr SelectionModel::
        set_parity(Signal::OperationDesc::Ptr o, Filters::Bandpass* src, SaveInside si)
{
    if (si != SaveInside_UNCHANGED)
        src->_save_inside = si == SaveInside_TRUE;
    src->updateChunkFilter ();
    return o;
}


template<>
Signal::OperationDesc::Ptr SelectionModel::
        set_parity(Signal::OperationDesc::Ptr o, Tools::Support::OperationOtherSilent* src, SaveInside si)
{
    if (si == SaveInside_UNCHANGED || si == SaveInside_TRUE) {
        return o;
    } else {
        return Signal::OperationDesc::Ptr( new Signal::OperationSetSilent( src->section() ));
    }
}


template<>
Signal::OperationDesc::Ptr SelectionModel::
        set_parity(Signal::OperationDesc::Ptr o, Signal::OperationSetSilent* src, SaveInside si)
{
    if (si == SaveInside_UNCHANGED || si == SaveInside_TRUE) {
        return o;
    } else {
        return Signal::OperationDesc::Ptr( new Tools::Support::OperationOtherSilent(
                src->affected_samples().spannedInterval() )
        );
    }
}


//template<>
//Signal::pOperation SelectionModel::
//        set_parity(Selections::Support::SplineFilter* src, SaveInside si)
//{
//    Selections::Support::SplineFilter* dst;
//    Signal::pOperation o( dst=new Selections::Support::SplineFilter( *src ));

//    if (si != SaveInside_UNCHANGED)
//        dst->_save_inside = si == SaveInside_TRUE;
//    return o;
//}


Signal::OperationDesc::Ptr SelectionModel::
        copy_selection(Signal::OperationDesc::Ptr o, SaveInside si)
{
    if (!o)
        return o;

#define TEST_TYPE(T) \
    do { \
        T* p = dynamic_cast<T*>( &*w ); \
        if (p) return set_parity(o, p, si); \
    } while(false)

    o = read1(o)->copy();

    Signal::OperationDesc::WritePtr w(o);

    TEST_TYPE(Filters::Ellipse);
    TEST_TYPE(Filters::Rectangle);
    TEST_TYPE(Filters::Bandpass);
    TEST_TYPE(Tools::Support::OperationOtherSilent);
    TEST_TYPE(Signal::OperationSetSilent);
//    TEST_TYPE(Selections::Support::SplineFilter);

    throw std::logic_error("SelectionModel::copy_selection(" + vartype(*o) + ") is not implemented");
}



} // namespace Tools
