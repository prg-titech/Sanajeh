#include "sanajeh_device_code.h"

AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;

float kDt = 0.02;
float kGravityConstant = 6.673e-05;
float kDampeningFactor = 0.05;

__device__ Body::Body(float px, float py, float vx, float vy, float m) {
    this.pos_x = px;
    this.pos_y = py;
    this.vel_x = vx;
    this.vel_y = vy;
    this.mass = m;
    this.force_x = 0.0;
    this.force_y = 0.0;
}

__device__ void Body::compute_force() {
    this.force_x = 0.0;
    this.force_y = 0.0;
    __pyallocator__.device_do(Body, Body.apply_force, this);
}

__device__ void Body::apply_force(Body other) {
    if (other != this) {
        float dx = this.pos_x - other.pos_x;
        float dy = this.pos_x - other.pos_y;
        float dist = math.sqrt(dx * dx + dy * dy);
        float f = kGravityConstant * this.mass * other.mass / dist * dist + kDampeningFactor;
        other.force_x += f * dx / dist;
        other.force_y += f * dy / dist;
    }
}

__device__ void Body::body_update() {
    this.vel_x += this.force_x * kDt / this.mass;
    this.vel_y += this.force_y * kDt / this.mass;
    this.pos_x += this.vel_x * kDt;
    this.pos_y += this.vel_y * kDt;
    if (this.pos_x < -1 || this.pos_x > 1) {
        this.vel_x = -this.vel_x;
    }
    if (this.pos_y < -1 || this.pos_y > 1) {
        this.vel_y = -this.vel_y;
    }
}
