#ifndef SANAJEH_DEVICE_CODE_H
#define SANAJEH_DEVICE_CODE_H
#define KNUMOBJECTS 64*64*64*64

#include <curand_kernel.h>
#include "dynasoar.h"

class Cell;
class Agent;
class Fish;
class Shark;


using AllocatorT = SoaAllocator<KNUMOBJECTS, Cell, Agent, Fish, Shark>;

static const int kSeed = 42;
static const int kSpawnThreshold = 4;
static const bool kOptionSharkDie = true;
static const bool kOptionFishSpawn = true;
static const int kEnergyBoost = 4;
static const int kEnergyStart = 2;
static const int kSizeX = 100;
static const int kSizeY = 100;

class Cell : public AllocatorT::Base {
	public:
		declare_field_types(Cell, curandState, DeviceArray<Cell*, 4>, Agent*, DeviceArray<bool, 5>, int)
		Field<Cell, 0> random_state_;
		Field<Cell, 1> neighbors_;
		Field<Cell, 2> agent_ref;
		Field<Cell, 3> neighbor_request_;
		Field<Cell, 4> id;
	
		__device__ Cell();
		__device__ Cell(int cell_id);
		__device__ void setup();
		__device__ Agent* agent();
		__device__ void decide();
		__device__ void enter(Agent* agent);
		__device__ bool has_fish();
		__device__ bool has_shark();
		__device__ bool is_free();
		__device__ void kill();
		__device__ void leave();
		__device__ void prepare();
		__device__ void set_neighbors(Cell* left, Cell* top, Cell* right, Cell* bottom);
		__device__ void request_random_fish_neighbor();
		__device__ void request_random_free_neighbor();
		__device__ bool request_random_neighbor_has_fish(curandState& random_state);
		__device__ bool request_random_neighbor_is_free(curandState& random_state);
		void _do(void (*pf)(int, int, int, int, int));
};

class Agent : public AllocatorT::Base {
	public:
		declare_field_types(Agent, curandState, Cell*, Cell*, bool)
		Field<Agent, 0> random_state_;
		Field<Agent, 1> position_ref;
		Field<Agent, 2> new_position_ref;
		Field<Agent, 3> kIsAbstract;
	
		__device__ Agent();
		__device__ Agent(int seed);
		__device__ Cell* position();
		__device__ void set_new_position(Cell* new_pos);
		__device__ void set_position(Cell* cell);
		void _do(void (*pf)(int, int, int, bool));
};

class Fish : public Agent {
	public:
		declare_field_types(Fish, int, bool)

		using BaseClass = Agent;

		Field<Fish, 0> egg_timer_;
		Field<Fish, 1> kIsAbstract;
	
		__device__ Fish();
		__device__ Fish(int seed);
		__device__ void prepare();
		__device__ void update();
		void _do(void (*pf)(int, bool));
};

class Shark : public Agent {
	public:
		declare_field_types(Shark, int, int, bool)

		using BaseClass = Agent;

		Field<Shark, 0> egg_timer_;
		Field<Shark, 1> energy_;
		Field<Shark, 2> kIsAbstract;
	
		__device__ Shark();
		__device__ Shark(int seed);
		__device__ void prepare();
		__device__ void update();
		void _do(void (*pf)(int, int, bool));
};

extern "C" int Cell_do_all(void (*pf)(int, int, int, int, int));
extern "C" int Cell_Cell_setup();
extern "C" int Cell_Cell_prepare();
extern "C" int Fish_Fish_prepare();
extern "C" int Cell_Cell_decide();
extern "C" int Fish_Fish_update();
extern "C" int Shark_Shark_prepare();
extern "C" int Shark_Shark_update();
extern "C" int parallel_new_Cell(int object_num);
extern "C" int AllocatorInitialize();

extern "C" int AllocatorUninitialize();

#endif