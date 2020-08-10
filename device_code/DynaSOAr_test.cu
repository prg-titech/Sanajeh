#include <chrono>
#include "sanajeh_device_code.cu"

int main(int argc, char* argv[]) {
#ifdef OPTION_RENDER

#endif  // OPTION_RENDER
	AllocatorHandle<AllocatorT>* allocator_handler;
	__device__ AllocatorT* device_allocatorr;
  // Create new allocator.
	allocator_handler = new AllocatorHandle<AllocatorT>(/* unified_memory= */ true);
	AllocatorT* dev_ptr = allocator_handler->device_pointer();
	cudaMemcpyToSymbol(device_allocatorr, &dev_ptr, sizeof(AllocatorT*), 0, cudaMemcpyHostToDevice);

	allocator_handler->parallel_new<Body>(argv[1]);


	for (int i = 0; i < 100; ++i) {
	auto time_start = std::chrono::system_clock::now();
	allocator_handler->parallel_do<Body, &Body::compute_force>();
	allocator_handler->parallel_do<Body, &Body::body_update>();
	auto time_end = std::chrono::system_clock::now();
	auto elapsed = time_end - time_start;
	auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("%lu\n", micros);
	}

	return 0;
}