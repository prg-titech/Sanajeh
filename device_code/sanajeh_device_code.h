#ifndef SANAJEH_DEVICE_CODE_H
#define SANAJEH_DEVICE_CODE_H
#define KNUMOBJECTS 64*64*64*64

#include <curand_kernel.h>
#include "allocator_config.h"

class Body;

using AllocatorT = SoaAllocator<KNUMOBJECTS, Body>;

class Body {
    public:
    	declare_field_types(Body, float, float, float, float, float, float, float)
    private:
		Field<Body, 0> pos_x;
		Field<Body, 1> pos_y;
		Field<Body, 2> vel_x;
		Field<Body, 3> vel_y;
		Field<Body, 4> force_x;
		Field<Body, 5> force_y;
		Field<Body, 6> mass;

    __device__ Body(float px, float py, float vx, float vy, float m);
    __device__ void compute_force();
    __device__ void apply_force(Body other);
    __device__ void body_update();
};

#endif