#ifndef LOG_H
#define LOG_H

#include <boost/format.hpp>
#include <boost/noncopyable.hpp>

/**
 * @brief The Log class should make it easy to add a type-safe well-formatted log entry.
 */
class Log: public boost::format, private boost::noncopyable {
public:
    Log(const std::string&);
    ~Log();
};

#endif // LOG_H
