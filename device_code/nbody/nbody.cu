#include "nbody.h"

AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;

__device__ Body::Body(float px, float py, float vx, float vy, float fx, float fy, float m) {
	this->pos_x = px;
	this->pos_y = py;
	this->vel_x = vx;
	this->vel_y = vy;
	this->force_x = fx;
	this->force_y = fy;
	this->mass = m;
}

__device__ Body::Body(int idx) {
	curandState rand_state;
	curand_init(kSeed, idx, 0, &rand_state);
	this->pos_x = (2.0 * curand_uniform(&rand_state)) - 1.0;
	this->pos_y = (2.0 * curand_uniform(&rand_state)) - 1.0;
	this->vel_x = 0.0;
	this->vel_y = 0.0;
	this->force_x = 0.0;
	this->force_y = 0.0;
	this->mass = ((curand_uniform(&rand_state) / 2.0) + 0.5) * kMaxMass;
}

__device__ void Body::compute_force() {
	this->force_x = 0.0;
	this->force_y = 0.0;
	device_allocator->template device_do<Body>(&Body::apply_force, this);
}

__device__ void Body::apply_force(Body* other) {
	if (other != this) {
		float d_x = this->pos_x - other->pos_x;
		float d_y = this->pos_y - other->pos_y;
		float dist = sqrt((d_x * d_x) + (d_y * d_y));
		float f = ((kGravityConstant * this->mass) * other->mass) / ((dist * dist) + kDampeningFactor);
		float __auto_v0_x = d_x * f;
		float __auto_v0_y = d_y * f;
		float __auto_v1_x = __auto_v0_x / dist;
		float __auto_v1_y = __auto_v0_y / dist;
		other->force_x += __auto_v1_x;
		other->force_y += __auto_v1_y;
	}
}

__device__ void Body::body_update() {
	float __auto_v0_x = this->force_x * kDt;
	float __auto_v0_y = this->force_y * kDt;
	float __auto_v1_x = __auto_v0_x / this->mass;
	float __auto_v1_y = __auto_v0_y / this->mass;
	this->vel_x += __auto_v1_x;
	this->vel_y += __auto_v1_y;
	float __auto_v2_x = this->vel_x * kDt;
	float __auto_v2_y = this->vel_y * kDt;
	this->pos_x += __auto_v2_x;
	this->pos_y += __auto_v2_y;
	if (this->pos_x < -1 || this->pos_x > 1) {
		this->vel_x = -this->vel_x;
	}
	if (this->pos_y < -1 || this->pos_y > 1) {
		this->vel_y = -this->vel_y;
	}
}

void Body::_do(void (*pf)(float, float, float, float, float, float, float)){
	pf(this->pos_x, this->pos_y, this->vel_x, this->vel_y, this->force_x, this->force_y, this->mass);
}

extern "C" int Body_do_all(void (*pf)(float, float, float, float, float, float, float)){
	allocator_handle->template device_do<Body>(&Body::_do, pf);
 	return 0;
}

extern "C" int Body_Body_compute_force(){
	allocator_handle->parallel_do<Body, &Body::compute_force>();
	return 0;
}

extern "C" int Body_Body_body_update(){
	allocator_handle->parallel_do<Body, &Body::body_update>();
	return 0;
}

extern "C" int parallel_new_Body(int object_num){
	allocator_handle->parallel_new<Body>(object_num);
	return 0;
}

extern "C" int AllocatorInitialize(){
	allocator_handle = new AllocatorHandle<AllocatorT>(/* unified_memory= */ true);
	AllocatorT* dev_ptr = allocator_handle->device_pointer();
	cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0, cudaMemcpyHostToDevice);
	return 0;
}

extern "C" int AllocatorUninitialize(){
	return 0;
}