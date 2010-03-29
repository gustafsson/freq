#ifndef SIGNALINVALIDSAMPLESDESCRIPTOR_H
#define SIGNALINVALIDSAMPLESDESCRIPTOR_H

namespace Signal {

class InvalidSamplesDescriptor
{
public:
    InvalidSamplesDescriptor();
    InvalidSamplesDescriptor& operator |= (const InvalidSamplesDescriptor& b);
};

} // namespace Signal

#endif // SIGNALINVALIDSAMPLESDESCRIPTOR_H
