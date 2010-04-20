#ifndef SAMPLESINTERVALDESCRIPTOR_H
#define SAMPLESINTERVALDESCRIPTOR_H

#include <list>

namespace Signal {

class SamplesIntervalDescriptor
{
public:
    struct Interval {
        float first, last;

        bool operator<(const Interval& r) const;
        bool operator|=(const Interval& r);
    };

    SamplesIntervalDescriptor();
    SamplesIntervalDescriptor(float first, float last);
    SamplesIntervalDescriptor& operator |= (const SamplesIntervalDescriptor&);
    SamplesIntervalDescriptor& operator |= (const Interval&);

    Interval    popInterval( float dt, float center = 0 );
    void        makeValid( Interval );

private:
    std::list<Interval> _intervals;
};

} // namespace Signal

#endif // SAMPLESINTERVALDESCRIPTOR_H
