#include <chrono>
#include "sanajeh_device_code.cu"

int main(int argc, char* argv[]) {
#ifdef OPTION_RENDER

#endif  // OPTION_RENDER

  // Create new allocator.
  AllocatorInitialize()

  allocator_handle->parallel_new<Body>(argv[1]);


  for (int i = 0; i < 100; ++i) {
	auto time_start = std::chrono::system_clock::now();
    allocator_handle->parallel_do<Body, &Body::compute_force>();
    allocator_handle->parallel_do<Body, &Body::update>();
	auto time_end = std::chrono::system_clock::now();
	auto elapsed = time_end - time_start;
	auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("%lu\n", micros);
  }

  return 0;
}