#include "layer.h"

namespace Signal {

class OperationAddChannels: public Operation
{
public:
    OperationAddChannels( pOperation source, pOperation source2 );

    virtual pBuffer read( const Interval& I );

    virtual pOperation source2() const { return _source2; }

private:
    pOperation _source2;
};


Layers::Layers()
{
}

} // namespace Signal
