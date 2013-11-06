#ifndef PRETTIFYSEGFAULT_H
#define PRETTIFYSEGFAULT_H

#include <boost/exception/all.hpp>

/**
 * @brief The PrettifySegfault class should attempt to capture any null-pointer
 * exception in the program and throw a controlled C++ exception instead from
 * that location instead.
 *
 *     When you attempt to recover from segfaults,
 *     you are playing with fire.
 *
 *     Once a segfault has been detected without a crash,
 *     you should restart the process. It is likely that a segfault
 *     will still crash the process, but there should at least be
 *     some info in the log.
 *
 * Throwing from within a signal handler is undefined behavior. There is a
 * proposed solution to unwind from null-pointer exceptions at:
 * http://feepingcreature.github.io/handling.html
 * Looks neat, however it only works in special cases. See note number 4.
 * PrettifySegfault::test lists some different scenarios. The setup method
 * uses different handlers for sigsegv and other signals.
 *
 * While at it, PrettifySegfault may attempt to log and throw exceptions for
 * other types of signals as well if they are detected.
 */
class PrettifySegfault
{
public:
    enum SignalHandlingState {
        normal_execution,
        doing_signal_handling
    };

    /**
     * @brief setup enables PrettifySegfault.
     */
    static void setup();

    /**
     * @return If the process is in the state of signal handling you
     * should proceed to exit the process.
     */
    static SignalHandlingState signal_handling_state ();
    static bool has_caught_any_signal ();

    /**
     * @brief PrettifySegfaultDirectPrint makes the signal handler write info
     * to stdout as soon as the signal is caught. Default enable=true.
     * @param enable
     */
    static void EnableDirectPrint(bool enable);

    /**
     * @brief test will cause a segfault. This will put the process in
     * signal_handling_state and prevent further signals from being caught.
     * Such signals will instead halt the process.
     */
    static void test();
};


class signal_exception: virtual public boost::exception, virtual public std::exception {
public:
    typedef boost::error_info<struct signal, int> signal;
    typedef boost::error_info<struct signalname, const char*> signalname;
    typedef boost::error_info<struct signaldesc, const char*> signaldesc;
};
class segfault_exception: public signal_exception {};

#endif // PRETTIFYSEGFAULT_H
