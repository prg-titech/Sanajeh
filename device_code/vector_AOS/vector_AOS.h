#ifndef SANAJEH_DEVICE_CODE_H
#define SANAJEH_DEVICE_CODE_H
#define KNUMOBJECTS 64*64*64*64

#include <curand_kernel.h>
#include "dynasoar.h"

class Vector;
class Body;


using AllocatorT = SoaAllocator<KNUMOBJECTS, Vector, Body>;

static const int kSeed = 45;
static const float kMaxMass = 1000.0;
static const float kDt = 0.01;
static const float kGravityConstant = 4e-06;
static const float kDampeningFactor = 0.05;

class Vector{
	public:
		declare_field_types(Vector, float, float)
		Field<Vector, 0> x;
		Field<Vector, 1> y;

		__device__ Vector(float x_, float y_);
		__device__ Vector(int idx);
		__device__ Vector* add(Vector* other);
		__device__ Vector plus(Vector* other);
		__device__ Vector* subtract(Vector* other);
		__device__ Vector minus(Vector* other);
		__device__ Vector* scale(float ratio);
		__device__ Vector multiply(float multiplier);
		__device__ Vector* divide_by(float divisor);
		__device__ Vector divide(float divisor);
		__device__ float dist_origin();
		__device__ Vector* to_zero();
		void _do(void (*pf)());
};

class Body : public AllocatorT::Base {
	public:
		declare_field_types(Body, float, Vector, Vector, Vector)
		Field<Body, 0> mass;
		Field<Body, 1> pos;
		Field<Body, 2> vel;
		Field<Body, 3> force;


		__device__ Body(float px, float py, float vx, float vy, float fx, float fy, float m);
		__device__ Body(int idx);
		__device__ void compute_force();
		__device__ void apply_force(Body* other);
		__device__ void body_update();
};

extern "C" int Body_Body_compute_force();
extern "C" int Body_Body_body_update();
extern "C" int parallel_new_Body(int object_num);
extern "C" int parallel_new_Vector(int object_num);
extern "C" int AllocatorInitialize();

extern "C" int AllocatorUninitialize();

#endif
