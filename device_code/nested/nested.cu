#include "nested.h"

AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;

__device__ Vector::Vector(float x_, float y_) {
	float this->x = x_;
	float this->y = y_;
}

__device__ Vector* Vector::add(Vector* other) {
	this->x += other->x;
	this->y += other->y;
	return this;
}

__device__ Vector* Vector::plus(Vector* other) {
	return Vector(this->x + other->x, this->y + other->y);
}

__device__ Vector* Vector::subtract(Vector* other) {
	this->x -= other->x;
	this->y -= other->y;
	return this;
}

__device__ Vector* Vector::minus(Vector* other) {
	return Vector(this->x - other->x, this->y - other->y);
}

__device__ Vector* Vector::scale(float ratio) {
	this->x *= ratio;
	this->y *= ratio;
	return this;
}

__device__ Vector* Vector::multiply(float multiplier) {
	return Vector(this->x * multiplier, this->y * multiplier);
}

__device__ Vector* Vector::divide_by(float divisor) {
	this->x /= divisor;
	this->y /= divisor;
	return this;
}

__device__ Vector* Vector::divide(float divisor) {
	return Vector(this->x / divisor, this->y / divisor);
}

__device__ float Vector::dist_origin() {
	return sqrt((this->x * this->x) + (this->y * this->y));
}

__device__ Vector* Vector::to_zero() {
	this->x = 0.0;
	this->y = 0.0;
	return this;
}

__device__ Body::Body(float px, float py, float vx, float vy, float fx, float fy, float m) {
	this->pos = Vector(px, py);
	this->vel = Vector(vx, vy);
	this->force = Vector(fx, fy);
	this->mass = m;
}

__device__ Body::Body(int idx) {
	curandState rand_state;
	curand_init(kSeed, idx, 0, &rand_state);
	this->pos = Vector((2.0 * curand_uniform(&rand_state)) - 1.0, (2.0 * curand_uniform(&rand_state)) - 1.0);
	this->vel = Vector(0.0, 0.0);
	this->force = Vector(0.0, 0.0);
	this->mass = ((curand_uniform(&rand_state) / 2.0) + 0.5) * kMaxMass;
}

__device__ void Body::compute_force() {
	this->force->to_zero();
	device_allocator->template device_do<Body>(&Body::apply_force, this);
}

__device__ void Body::apply_force(Body* other) {
	if (other != this) {
		Vector* d = this->pos->minus(other->pos);
		float dist = d->dist_origin();
		float f = ((kGravityConstant * this->mass) * other->mass) / ((dist * dist) + kDampeningFactor);
		other->force->add(d->multiply(f)->divide(dist));
	}
}

__device__ void Body::body_update() {
	this->vel->add(this->force->multiply(kDt)->divide(this->mass));
	this->pos->add(this->vel->multiply(kDt));
	if (this->pos->x < -1 || this->pos->x > 1) {
		this->vel->x = -this->vel->x;
	}
	if (this->pos->y < -1 || this->pos->y > 1) {
		this->vel->y = -this->vel->y;
	}
}

void Body::_do(void (*pf)(Vector*, Vector*, Vector*, float)){
	pf(this->pos, this->vel, this->force, this->mass);
}

extern "C" int Body_do_all(void (*pf)(Vector*, Vector*, Vector*, float)){
	allocator_handle->template device_do<Body>(&Body::_do, pf);
 	return 0;
}


void Vector::_do(void (*pf)()){
	pf();
}

extern "C" int Vector_do_all(void (*pf)()){
	allocator_handle->template device_do<Vector>(&Vector::_do, pf);
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

extern "C" int parallel_new_Vector(int object_num){
	allocator_handle->parallel_new<Vector>(object_num);
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