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
    Chain::WritePtr chain(chain_);
    IInvalidator::Ptr i = chain->addOperationAt ( operation_, at_ );

    i=i; // discard IInvalidator and avoid compiler warning about it
}


void AppendOperationDescCommand::
        undo()
{
    write1(chain_)->removeOperationsAt ( at_ );
}


std::string AppendOperationDescCommand::
        toString()
{
    return operation_->toString ().toStdString ();
}


} // namespace Commands
} // namespace Tools


// Unit test
#include "test/operationmockups.h"

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
    // It should add a new operation to the signal processing chain at the given targets current position
    {
        Chain::Ptr chain = Chain::createDefaultChain ();
        OperationDesc::Ptr target_desc(new Test::TransparentOperationDesc);
        OperationDesc::Ptr operation_desc(new Test::TransparentOperationDesc);
        OperationDesc::Ptr source_desc(new SourceMock);
        TargetMarker::Ptr target = write1(chain)->addTarget(target_desc);


        AppendOperationDescCommand aodc1(source_desc, chain, target);
        AppendOperationDescCommand aodc2(operation_desc, chain, target);


        EXCEPTION_ASSERT( !read1(chain)->extent (target).interval.is_initialized() );

        aodc1.execute ();
        EXCEPTION_ASSERT_EQUALS( read1(chain)->extent (target).interval, Interval(3,5));

        aodc1.undo ();
        EXCEPTION_ASSERT( !read1(chain)->extent (target).interval.is_initialized() );

        aodc1.execute ();
        aodc2.execute ();
        EXCEPTION_ASSERT_EQUALS( read1(chain)->extent (target).interval, Interval(3,5));

        aodc2.undo ();
        EXCEPTION_ASSERT_EQUALS( read1(chain)->extent (target).interval, Interval(3,5));

        aodc1.undo ();
        EXCEPTION_ASSERT( !read1(chain)->extent (target).interval.is_initialized() );
    }
}

} // namespace Commands
} // namespace Tools
