#include "appendoperationdesccommand.h"

using namespace Signal;
using namespace Processing;

namespace Tools {
namespace Commands {

AppendOperationDescCommand::
        AppendOperationDescCommand(OperationDesc::ptr o,
                                   Chain::ptr c,
                                   TargetMarker::ptr a)
    :
      operation_(o),
      chain_(c),
      at_(a)
{
}


void AppendOperationDescCommand::
        execute()
{
    IInvalidator::ptr i = chain_->addOperationAt ( operation_, at_ );

    operation_.write ()->setInvalidator (i);
}


void AppendOperationDescCommand::
        undo()
{
    chain_->removeOperationsAt ( at_ );
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
#include <QtWidgets> // QApplication

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
        Chain::ptr chain = Chain::createDefaultChain ();
        OperationDesc::ptr target_desc(new Test::TransparentOperationDesc);
        OperationDesc::ptr operation_desc(new Test::TransparentOperationDesc);
        OperationDesc::ptr source_desc(new SourceMock);
        TargetMarker::ptr target = chain->addTarget(target_desc);


        AppendOperationDescCommand aodc1(source_desc, chain, target);
        AppendOperationDescCommand aodc2(operation_desc, chain, target);


        EXCEPTION_ASSERT( !chain->extent (target).interval.is_initialized() );

        aodc1.execute ();
        EXCEPTION_ASSERT_EQUALS( chain->extent (target).interval, Interval(3,5));

        aodc1.undo ();
        EXCEPTION_ASSERT( !chain->extent (target).interval.is_initialized() );

        aodc1.execute ();
        aodc2.execute ();
        EXCEPTION_ASSERT_EQUALS( chain->extent (target).interval, Interval(3,5));

        aodc2.undo ();
        EXCEPTION_ASSERT_EQUALS( chain->extent (target).interval, Interval(3,5));

        aodc1.undo ();
        EXCEPTION_ASSERT( !chain->extent (target).interval.is_initialized() );
    }
}

} // namespace Commands
} // namespace Tools
