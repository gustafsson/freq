#include "operationwrapper.h"

#include <boost/foreach.hpp>
#include <boost/weak_ptr.hpp>

namespace Signal {

OperationWrapper::
        OperationWrapper(Operation::Ptr wrap)
{
    setWrappedOperation(wrap);
}


void OperationWrapper::
        setWrappedOperation(Operation::Ptr wrap)
{
    wrap_ = wrap;
}


Signal::pBuffer OperationWrapper::
        process(Signal::pBuffer b)
{
    if (!wrap_)
        return b;

    return wrap_->process (b);
}


OperationDescWrapper::
        OperationDescWrapper(OperationDesc::Ptr wrap)
    :
      map_(new OperationMap())
{
    setWrappedOperationDesc(wrap);
}


void OperationDescWrapper::
        setWrappedOperationDesc(OperationDesc::Ptr wrap)
{
    BOOST_FOREACH(OperationMap::value_type& v, *map_)
    {
        Operation::Ptr o = v.second.lock();
        if (o) {
            OperationWrapper* w = dynamic_cast<OperationWrapper*>(o.get());
            EXCEPTION_ASSERT(w);

            Operation::Ptr o;
            if (wrap)
                o = wrap->createOperation (v.first);

            w->setWrappedOperation (o);
        }
    }

    wrap_ = wrap;
}


OperationDesc::Ptr OperationDescWrapper::
        getWrappedOperationDesc()
{
    return wrap_;
}


Signal::Interval OperationDescWrapper::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (wrap_)
        return wrap_->requiredInterval (I, expectedOutput);

    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Signal::Interval OperationDescWrapper::
        requiredInterval( const Signal::Intervals& I, Signal::Interval* expectedOutput ) const
{
    if (wrap_)
        return wrap_->requiredInterval (I, expectedOutput);

    return OperationDesc::requiredInterval (I, expectedOutput);
}


Interval OperationDescWrapper::
        affectedInterval( const Interval& I ) const
{
    if (wrap_)
        return wrap_->affectedInterval (I);
    return I;
}


Intervals OperationDescWrapper::
        affectedInterval( const Intervals& I ) const
{
    if (wrap_)
        return wrap_->affectedInterval (I);
    return I;
}


OperationDesc::Ptr OperationDescWrapper::
        copy() const
{
    OperationDesc::Ptr od(new OperationDescWrapper(wrap_));
    return od;
}


Operation::Ptr OperationDescWrapper::
        createOperation(ComputingEngine* engine=0) const
{
    Operation::Ptr o;
    if (wrap_)
        o = wrap_->createOperation (engine);

    Operation::Ptr wo(createOperationWrapper(engine, o));
    (*map_)[engine] = wo;

    return wo;
}


OperationWrapper* OperationDescWrapper::
        createOperationWrapper(ComputingEngine*, Operation::Ptr wrapped) const
{
    return new OperationWrapper(wrapped);
}


OperationDesc::Extent OperationDescWrapper::
        extent() const
{
    if (wrap_)
        return wrap_->extent ();

    return Extent();
}


Operation::Ptr OperationDescWrapper::
        recreateOperation(Operation::Ptr, ComputingEngine*) const
{
    EXCEPTION_ASSERTX(false, "Not implemented");
}


QString OperationDescWrapper::
        toString() const
{
    if (wrap_)
        return wrap_->toString ();

    return vartype(*this).c_str ();
}


int OperationDescWrapper::
        getNumberOfSources() const
{
    EXCEPTION_ASSERTX(false, "Not implemented");
}


bool OperationDescWrapper::
        operator==(const OperationDesc& d) const
{
    const OperationDescWrapper* w = dynamic_cast<const OperationDescWrapper*>(&d);
    if (!w)
        return false;

    if (w == this)
        return true;

    if (!wrap_ && !w->wrap_)
        return true;

    if (wrap_ && w->wrap_)
        return *wrap_ == *w->wrap_;

    return false;
}


class EmptyOperationDesc : public OperationDesc
{
    Interval requiredInterval( const Interval& I, Interval* J) const {
        if (J) *J = I;
        return I;
    }

    Interval affectedInterval( const Interval& ) const {
        EXCEPTION_ASSERTX(false, "not implemented");
        return Interval();
    }

    OperationDesc::Ptr copy() const {
        EXCEPTION_ASSERTX(false, "not implemented");
        return OperationDesc::Ptr();
    }

    Operation::Ptr createOperation( ComputingEngine* ) const {
        return Operation::Ptr();
    }
};


class ExtentDesc : public EmptyOperationDesc
{
    Extent extent() const {
        Extent x;
        x.sample_rate = 17;
        return x;
    }
};


void OperationDescWrapper::
        test()
{
    // It should behave as another OperationDesc
    {
        OperationDescWrapper w(OperationDesc::Ptr(new ExtentDesc()));
        EXCEPTION_ASSERT_EQUALS( w.extent ().sample_rate.get_value_or (-1), 17);
    }
}

} // namespace Signal
