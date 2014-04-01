#include "selection.h"

namespace Filters {

Selection::
        ~Selection()
{
}


bool Selection::
        isExteriorSelected() const
{
    return !isInteriorSelected ();
}


void Selection::
        selectExterior(bool v)
{
    selectInterior(!v);
}

} // namespace Filters

#include "test/operationmockups.h"

namespace Filters {

class DummySelectionOperaitonTest: public Test::TransparentOperationDesc, public Selection {
    virtual bool isInteriorSelected() const { return select_interior; }
    virtual void selectInterior(bool v) { select_interior = v; }
    bool select_interior = true;
};

void Selection::
        test()
{
    // It should tag that the behaviour of a class can be flipped to select the
    // interior or the exterior part.
    {
        Signal::OperationDesc::Ptr o( new DummySelectionOperaitonTest );
        auto wo = o.write ();
        Selection* s = dynamic_cast<Selection*>(&*wo);

        EXCEPTION_ASSERT(!s->isExteriorSelected ());
        s->selectExterior ();
        EXCEPTION_ASSERT(s->isExteriorSelected ());
        s->selectExterior (false);
        EXCEPTION_ASSERT(s->isInteriorSelected ());
    }
}

} // namespace Filters
