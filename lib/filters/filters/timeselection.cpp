#include "timeselection.h"

#include "signal/operation-basic.h"
#include "filters/support/operation-composite.h"

using namespace Signal;
using namespace Tools::Support;

namespace Filters {

TimeSelection::
        TimeSelection(Interval section, bool select_interior)
    :
      section_(section)
{
    selectInterior (select_interior);
}


OperationDesc::ptr TimeSelection::
        copy() const
{
    return OperationDesc::ptr(new TimeSelection(section_));
}


bool TimeSelection::
        isInteriorSelected() const
{
    return dynamic_cast<OperationCrop*>(getWrappedOperationDesc ().raw ());
}


void TimeSelection::
        selectInterior(bool v)
{
    if (v)
      {
        this->setWrappedOperationDesc (OperationDesc::ptr(new OperationCrop(section_)));
      }
    else
      {
        this->setWrappedOperationDesc (OperationDesc::ptr(new OperationSetSilent(section_)));
      }
}


Interval TimeSelection::
        section() const
{
    return section_;
}

} // namespace Filters

#include "test/randombuffer.h"

namespace Filters {


void TimeSelection::
        test()
{
    // It should select samples within or outside of a section.
    {
        Interval section(5,7);

        pBuffer gold_exterior = Test::RandomBuffer::smallBuffer ();
        *gold_exterior |= Signal::Buffer(section, gold_exterior->sample_rate (), gold_exterior->number_of_channels ());

        pBuffer gold_interior = Test::RandomBuffer::smallBuffer ();
        foreach(Signal::Interval i, Signal::Intervals(gold_interior->getInterval ()) - section)
            *gold_interior |= Signal::Buffer(i, gold_interior->sample_rate (), gold_interior->number_of_channels ());

        TimeSelection ts(section);
        pBuffer b;

        Operation::ptr o = ts.createOperation (0);
        b = o->process (Test::RandomBuffer::smallBuffer ());
        EXCEPTION_ASSERT(*gold_interior == *b);

        ts.selectInterior (false);
        b = o->process (Test::RandomBuffer::smallBuffer ());
        EXCEPTION_ASSERT(*gold_interior == *b);

        o = ts.createOperation (0);
        b = o->process (Test::RandomBuffer::smallBuffer ());
        EXCEPTION_ASSERT(*gold_exterior == *b);
    }
}

} // namespace Filters
