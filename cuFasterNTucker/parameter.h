#include <numeric>
#include <iomanip>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <sys/time.h>

#define type_of_data float
#define grid_size 128
#define block_size 32
#define warp_size 32
#define sum_size 1024
#define error_size 1024

//#define mask 0xffffffff
#define mask 0x00000000

#define register_size 32

