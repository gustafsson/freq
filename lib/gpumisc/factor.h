#ifndef FACTOR_H
#define FACTOR_H

#include <vector>

/**
 * @brief The Factor class should factor a number 'n' into its prime factors.
 */
class Factor
{
public:
    typedef std::vector<unsigned> vector;

    static vector factor(unsigned n);

public:
    static void test();
};

#endif // FACTOR_H
