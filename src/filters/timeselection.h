#ifndef FILTERS_TIMESELECTION_H
#define FILTERS_TIMESELECTION_H

#include "selection.h"
#include "signal/operationwrapper.h"

namespace Filters {

/**
 * @brief The TimeSelection class should select samples within or outside of a section.
 */
class TimeSelection: public Signal::OperationDescWrapper, public Selection
{
public:
    TimeSelection(Signal::Interval section, bool select_interior=true);

    // Signal::OperationDesc
    OperationDesc::ptr copy() const override;

    // Selection
    bool isInteriorSelected() const override;
    void selectInterior(bool v) override;

    Signal::Interval section() const;

private:
    Signal::Interval section_;

public:
    static void test();
};

} // namespace Filters

#endif // FILTERS_TIMESELECTION_H
