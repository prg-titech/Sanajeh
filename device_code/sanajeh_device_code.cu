#include "sanajeh_device_code.h"

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
	this->vel_x = (curand_uniform(&rand_state) - 0.5) / 1000.0;
	this->vel_y = (curand_uniform(&rand_state) - 0.5) / 1000.0;
	this->mass = ((curand_uniform(&rand_state) / 2.0) + 0.5) * kMaxMass;
	this->force_x = 0.0;
	this->force_y = 0.0;
}

__device__ void Body::compute_force() {
	this->force_x = 0.0;
	this->force_y = 0.0;
	device_allocator->template device_do<Body>(&Body::apply_force, this);
}

__device__ void Body::apply_force(Body* other) {
	if (other != this) {
		float dx = this->pos_x - other->pos_x;
		float dy = this->pos_x - other->pos_y;
		float dist = sqrt((dx * dx) + (dy * dy));
		float f = ((kGravityConstant * this->mass) * other->mass) / ((dist * dist) + kDampeningFactor);
		other->force_x += (f * dx) / dist;
		other->force_y += (f * dy) / dist;
	}
}

__device__ void Body::body_update() {
	this->vel_x += (this->force_x * kDt) / this->mass;
	this->vel_y += (this->force_y * kDt) / this->mass;
	this->pos_x += this->vel_x * kDt;
	this->pos_y += this->vel_y * kDt;
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

int main(int argc, char* argv[]) {

	AllocatorInitialize();
	parallel_new_Body(atoi(argv[1]));
	long unsigned int overall = 0;

	for (int i = 0; i < 100; ++i) {
		auto time_start = std::chrono::system_clock::now();
		Body_Body_compute_force();
		Body_Body_body_update();
		auto time_end = std::chrono::system_clock::now();
		auto elapsed = time_end - time_start;
		auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	}
	printf("%lu\n", overall/100);

	return 0;
}