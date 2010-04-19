#ifndef SIGNALINVALIDSAMPLESDESCRIPTOR_H
#define SIGNALINVALIDSAMPLESDESCRIPTOR_H

namespace Signal {

class InvalidSamplesDescriptor
{
public:
    struct Interval {
        float first, last;

        bool operator<(const Range& r);
        bool operator|=(const Range& r);
    };

    InvalidSamplesDescriptor();
    InvalidSamplesDescriptor(float first, float last);
    InvalidSamplesDescriptor& operator |= (const InvalidSamplesDescriptor&);
    InvalidSamplesDescriptor& operator |= (const Interval&);

    Interval    popInterval( float dt, float center = 0 );
    void        makeValid( Interval );

private:
    std::list<Interval> _intervals;
};

} // namespace Signal

#endif // SIGNALINVALIDSAMPLESDESCRIPTOR_H
