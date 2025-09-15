#ifndef EXAMPLES_COMMON_DTYPES_H
#define EXAMPLES_COMMON_DTYPES_H

#include <float8.h>
#include <intn.h>
#include <mxfloat.h>

#include <Eigen/Core>
// float16 and bfloat16
using float16 = Eigen::half;
using bfloat16 = Eigen::bfloat16;

// 8-bit floating point representations, parameterized by number of exponent and
// mantissa bits, as well as the bias (if any) and representability of infinity,
// NaN, and signed zero.
using float8_e3m4 = ml_dtypes::float8_e3m4;
using float8_e4m3 = ml_dtypes::float8_e4m3;
using float8_e4m3b11fnuz = ml_dtypes::float8_e4m3b11fnuz;
using float8_e4m3fn = ml_dtypes::float8_e4m3fn;
using float8_e4m3fnuz = ml_dtypes::float8_e4m3fnuz;
using float8_e5m2 = ml_dtypes::float8_e5m2;
using float8_e5m2fnuz = ml_dtypes::float8_e5m2fnuz;
using float8_e8m0fnu = ml_dtypes::float8_e8m0fnu;

// Microscaling(MX) sub-byte floating point representations.
using float4_e2m1fn = ml_dtypes::float4_e2m1fn;
using float6_e2m3fn = ml_dtypes::float6_e2m3fn;
using float6_e3m2fn = ml_dtypes::float6_e3m2fn;

// Narrow integer encodings.
using int2 = ml_dtypes::int2;
using int4 = ml_dtypes::int4;
using uint2 = ml_dtypes::uint2;
using uint4 = ml_dtypes::uint4;

#endif
