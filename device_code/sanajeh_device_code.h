#ifndef SANAJEH_DEVICE_CODE_H
#define SANAJEH_DEVICE_CODE_H
#define KNUMOBJECTS 64*64*64*64

#include <curand_kernel.h>
#include "dynasoar.h"

class Body;

using AllocatorT = SoaAllocator<KNUMOBJECTS, Body>;

static const int kSeed = 3000;
static const float kMaxMass = 1000.0;
static const float kDt = 0.02;
static const float kGravityConstant = 6.673e-05;
static const float kDampeningFactor = 0.05;

class Body : public AllocatorT::Base {
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
	public:
		__device__ Body(int idx);
		__device__ void compute_force();
		__device__ void apply_force(Body* other);
		__device__ void body_update();
		void _do(void (*pf)(float, float, float, float, float, float, float));
};

extern "C" int Body_do_all(void (*pf)(float, float, float, float, float, float, float));
extern "C" int Body_Body_compute_force();
extern "C" int Body_Body_body_update();
extern "C" int parallel_new_Body(int object_num);
extern "C" int AllocatorInitialize();

#endif