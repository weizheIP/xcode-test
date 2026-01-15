#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" void MutiFun(unsigned char *a, float *b,int size,int depart);