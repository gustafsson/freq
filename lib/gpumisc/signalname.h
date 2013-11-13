#ifndef SIGNALNAME_H
#define SIGNALNAME_H

class SignalName
{
public:
    static const char* name(int signal);
    static const char* desc(int signal);
};

#endif // SIGNALNAME_H
