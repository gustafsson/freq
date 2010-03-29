#include "signal-sink.h"

namespace Signal {

Sink::Sink()
:   _expected_samples_left( 0 )
{}

unsigned Sink::expected_samples_left()
{
    return _expected_samples_left;
}

void Sink::expected_samples_left(unsigned expected_samples_left)
{
    _expected_samples_left = expected_samples_left;
}

} // namespace Signal
