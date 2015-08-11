#ifndef LOGTICKFREQUENCY_H
#define LOGTICKFREQUENCY_H

#include "timer.h"
#include <string>

class LogTickFrequency final
{
public:
    LogTickFrequency(std::string title="tickfrequency", double loginterval=10);
    ~LogTickFrequency();

    // return true if loginterval has passed
    // and if lognow is true then log is called
    bool tick(bool lognow=true);

    double hz(bool reset=true);
    void log();

private:
    std::string title_;
    double loginterval_;
    double T_;
    Timer timer_;
    int ticks_;
};

#endif // LOGTICKFREQUENCY_H
