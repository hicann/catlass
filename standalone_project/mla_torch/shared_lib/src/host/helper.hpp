#ifndef EXAMPLES_COMMON_HELPER_HPP
#define EXAMPLES_COMMON_HELPER_HPP

#include <iostream>
#include <acl/acl.h>
#include <runtime/rt_ffts.h>
#include "tiling/platform/platform_ascendc.h"

// Macro function for unwinding acl errors.
#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

// Macro function for unwinding rt errors.
#define RT_CHECK(status)                                                                     \
    do {                                                                                     \
        rtError_t error = status;                                                            \
        if (error != RT_ERROR_NONE) {                                                        \
            std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error << std::endl;   \
        }                                                                                    \
    } while (0)

int Gcd(int a, int b)
{
    return b == 0 ? a : Gcd(b, a % b);
}

int Lcm(int a, int b)
{
    return abs(a * b) / Gcd(a, b);
}

#endif  // EXAMPLES_COMMON_HELPER_HPP
