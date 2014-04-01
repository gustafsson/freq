#include "appendoperationdesccommand.h"

using namespace Signal;
using namespace Processing;

namespace Tools {
namespace Commands {

AppendOperationDescCommand::
        AppendOperationDescCommand(OperationDesc::Ptr o,
                                   Chain::Ptr c,
                                   TargetMarker::Ptr a)
    :
      operation_(o),
      chain_(c),
      at_(a)
{
}


void AppendOperationDescCommand::
        execute()
{
    IInvalidator::Ptr i = chain_.write ()->addOperationAt ( operation_, at_ );

    operation_.write ()->setInvalidator (i);
}


void AppendOperationDescCommand::
        undo()
{
    chain_.write ()->removeOperationsAt ( at_ );
}


std::string AppendOperationDescCommand::
        toString()
{
    return operation_.read ()->toString ().toStdString ();
}


} // namespace Commands
} // namespace Tools


// Unit test
#include "test/operationmockups.h"
#include <QApplication>

namespace Tools {
namespace Commands {

class SourceMock : public Test::TransparentOperationDesc
{
    Extent extent() const {
        Extent x;
        x.interval = Interval(3,5);
        x.number_of_channels = 2;
        x.sample_rate = 10;
        return x;
    }
};


void AppendOperationDescCommand::
        test()
{
    std::string name = "AppendOperationDescCommand";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);

    // It should add a new operation to the signal processing chain at the given targets current position
    {
        Chain::Ptr chain = Chain::createDefaultChain ();
        OperationDesc::Ptr target_desc(new Test::TransparentOperationDesc);
        OperationDesc::Ptr operation_desc(new Test::TransparentOperationDesc);
        OperationDesc::Ptr source_desc(new SourceMock);
        TargetMarker::Ptr target = chain.write ()->addTarget(target_desc);


        AppendOperationDescCommand aodc1(source_desc, chain, target);
        AppendOperationDescCommand aodc2(operation_desc, chain, target);


        EXCEPTION_ASSERT( !chain.read ()->extent (target).interval.is_initialized() );

        aodc1.execute ();
        EXCEPTION_ASSERT_EQUALS( chain.read ()->extent (target).interval, Interval(3,5));

        aodc1.undo ();
        EXCEPTION_ASSERT( !chain.read ()->extent (target).interval.is_initialized() );

        aodc1.execute ();
        aodc2.execute ();
        EXCEPTION_ASSERT_EQUALS( chain.read ()->extent (target).interval, Interval(3,5));

        aodc2.undo ();
        EXCEPTION_ASSERT_EQUALS( chain.read ()->extent (target).interval, Interval(3,5));

        aodc1.undo ();
        EXCEPTION_ASSERT( !chain.read ()->extent (target).interval.is_initialized() );
    }
}

} // namespace Commands
} // namespace Tools
