#include "operationwrapper.h"

#include "demangle.h"

#include <boost/foreach.hpp>
#include <boost/weak_ptr.hpp>

namespace Signal {

/**
 * @brief The OperationWrapper class should wrap all calls to another Operation.
 *
 * It should be a transparent no-op operation if the other operation is null.
 */
class NoopOperationWrapper: public Operation {
public:
    virtual Signal::pBuffer process(Signal::pBuffer b) { return b; }
};


OperationDescWrapper::
        OperationDescWrapper(OperationDesc::ptr wrap)
    :
      wrap_(wrap)
{
}


void OperationDescWrapper::
        setWrappedOperationDesc(OperationDesc::ptr wrap)
{
    if (wrap == wrap_)
        return;

    wrap_ = wrap;
}


OperationDesc::ptr OperationDescWrapper::
        getWrappedOperationDesc() const
{
    return wrap_;
}


Signal::Interval OperationDescWrapper::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (wrap_)
        return wrap_.read ()->requiredInterval (I, expectedOutput);

    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Interval OperationDescWrapper::
        affectedInterval( const Interval& I ) const
{
    if (wrap_)
        return wrap_.read ()->affectedInterval (I);
    return I;
}


OperationDesc::ptr OperationDescWrapper::
        copy() const
{
    OperationDesc::ptr od(new OperationDescWrapper(wrap_));
    return od;
}


Operation::ptr OperationDescWrapper::
        createOperation(ComputingEngine* engine=0) const
{
    if (!wrap_)
        return Operation::ptr(new NoopOperationWrapper);

    return wrap_.read ()->createOperation (engine);
}


OperationDesc::Extent OperationDescWrapper::
        extent() const
{
    if (wrap_)
        return wrap_.read ()->extent ();

    return Extent();
}


QString OperationDescWrapper::
        toString() const
{
    if (wrap_)
        return wrap_.read ()->toString ();

    return vartype(*this).c_str ();
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
        return *wrap_.read () == *w->wrap_.read ();

    return false;
}


} // namespace Signal

#include "unused.h"
#include "timer.h"
#include "signal/computingengine.h"

#include <QSemaphore>
#include <QThread>

namespace Signal {

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

    OperationDesc::ptr copy() const {
        EXCEPTION_ASSERTX(false, "not implemented");
        return OperationDesc::ptr();
    }

    Operation::ptr createOperation( ComputingEngine* ) const {
        return Operation::ptr();
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

class WrappedOperationMock : public Signal::Operation {
    virtual Signal::pBuffer process(Signal::pBuffer b) {
        QSemaphore().tryAcquire (1, 2); // sleep 2 ms
        return b;
    }
};

class WrappedOperationDescMock : public EmptyOperationDesc
{
public:
    boost::shared_ptr<int> createOperationCount = boost::shared_ptr<int>(new int(0));

    Operation::ptr createOperation( ComputingEngine* ) const {
        (*createOperationCount)++;
        return Operation::ptr(new WrappedOperationMock);
    }
};

class WrappedOperationWorkerMock : public QThread
{
public:
    Operation::ptr operation;
    Signal::pBuffer buffer;

    void run () {
        if (operation && buffer)
            buffer = operation.write ()->process(buffer);
    }
};

void OperationDescWrapper::
        test()
{
    // It should behave as another OperationDesc
    {
        OperationDescWrapper w(OperationDesc::ptr(new ExtentDesc()));
        EXCEPTION_ASSERT_EQUALS( w.extent ().sample_rate.get_value_or (-1), 17);
    }

    // It should recreate instantiated operations when the wrapped operation is changed
    {
        OperationDesc::ptr a(new WrappedOperationDescMock());
        OperationDesc::ptr b(new WrappedOperationDescMock());
        OperationDescWrapper w(a);

        {
            EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*a.write ())->createOperationCount, 0 );
            EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*b.write ())->createOperationCount, 0 );
            UNUSED(Signal::Operation::ptr o1) = w.createOperation (0);
            EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*a.write ())->createOperationCount, 1 );
            EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*b.write ())->createOperationCount, 0 );
            w.setWrappedOperationDesc (b);
            EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*a.write ())->createOperationCount, 1 );
            EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*b.write ())->createOperationCount, 0 );
        }
        w.setWrappedOperationDesc (a);
        EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*a.write ())->createOperationCount, 1 );
        EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*b.write ())->createOperationCount, 0 );
    }

    // It should behave as a transparent operation if no operation is wrapped
    {
        OperationDescWrapper w;
        OperationDesc::ptr a(new WrappedOperationDescMock());

        Signal::Operation::ptr o0 = w.createOperation (0);
        EXCEPTION_ASSERT(o0);
        EXCEPTION_ASSERT_EQUALS(vartype(*o0.get ()), "Signal::NoopOperationWrapper");

        w.setWrappedOperationDesc (a);
        Signal::Operation::ptr o1 = w.createOperation (0);
        EXCEPTION_ASSERT(o1);
        EXCEPTION_ASSERT_EQUALS(vartype(*o1.get ()), "Signal::WrappedOperationMock");

        Signal::pBuffer b1(new Signal::Buffer(Signal::Interval(0,1),1,1));
        Signal::pBuffer b2(new Signal::Buffer(Signal::Interval(0,1),1,1));
        *b1->getChannel (0)->waveform_data ()->getCpuMemory () = 1234;
        *b2->getChannel (0)->waveform_data () = *b1->getChannel (0)->waveform_data ();
        Signal::pBuffer r1 = o1.write ()->process(b1);
        EXCEPTION_ASSERT_EQUALS(r1, b1);
        EXCEPTION_ASSERT(*r1 == *b2);
    }

    // It should allow new opertions without blocking while processing existing operations
    {
        OperationDescWrapper w;
        OperationDesc::ptr a(new WrappedOperationDescMock());
        OperationDesc::ptr b(new WrappedOperationDescMock());
        OperationDesc::ptr c(new WrappedOperationDescMock());

        WrappedOperationWorkerMock wowm;
        wowm.operation = w.createOperation (0);
        wowm.buffer = Signal::pBuffer(new Signal::Buffer(Signal::Interval(0,1),1,1));

        EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*a.write ())->createOperationCount, 0 );
        w.setWrappedOperationDesc (a);
        EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*a.write ())->createOperationCount, 0 );
        wowm.operation = w.createOperation (0);
        EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*a.write ())->createOperationCount, 1 );
        ComputingEngine::ptr ce(new ComputingCpu);
        wowm.operation = w.createOperation (ce.get());
        EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*a.write ())->createOperationCount, 2 );
        w.setWrappedOperationDesc (b);
        EXCEPTION_ASSERT_EQUALS( *dynamic_cast<WrappedOperationDescMock*>(&*a.write ())->createOperationCount, 2 );

        wowm.start ();
        QSemaphore().tryAcquire (1, 1); // sleep 1 ms to make sure the operation is locked for writing
        {
            Timer t;
            w.setWrappedOperationDesc (c); // this should not be blocking
            float T = t.elapsed ();
            EXCEPTION_ASSERT_LESS( T, 60e-6 );
        }
        {
            Timer t;
            EXCEPTION_ASSERT( wowm.wait (2) );
            float T = t.elapsed ();
            EXCEPTION_ASSERT_LESS( 0.5e-3, T ); // Should have waited at least 0.5 ms for processing to complete
        }
    }
}

} // namespace Signal
