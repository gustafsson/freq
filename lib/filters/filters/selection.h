#ifndef FILTERS_SELECTION_H
#define FILTERS_SELECTION_H

namespace Filters {

/**
 * @brief The Selection class should tag that the behaviour of a class can be
 * flipped to select the interior or the exterior part.
 */
class Selection
{
public:
    virtual ~Selection();

    virtual bool isInteriorSelected() const = 0;
    virtual void selectInterior(bool v=true) = 0;
    bool         isExteriorSelected() const;
    void         selectExterior(bool v=true);

public:
    static void test();
};

} // namespace Filters

#endif // FILTERS_SELECTION_H
