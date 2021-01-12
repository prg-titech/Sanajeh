#ifndef SANAJEH_DEVICE_CODE_H
#define SANAJEH_DEVICE_CODE_H
#define KNUMOBJECTS 64*64*64*64

#include <curand_kernel.h>
#include "dynasoar.h"

class Cell;
class Agent;
class Alive;
class Candidate;


using AllocatorT = SoaAllocator<KNUMOBJECTS, Cell, Agent, Alive, Candidate>;

static const int SIZE_X = 100;
static const int SIZE_Y = 100;

class Cell : public AllocatorT::Base {
	public:
		declare_field_types(Cell, Agent*)
		Field<Cell, 0> agent_;
	
		__device__ Cell(int idx);
		__device__ Agent* agent();
		__device__ bool is_empty();
		void _do(void (*pf)(Agent*));
};

class Agent : public AllocatorT::Base {
	public:
		declare_field_types(Agent, int, int, bool)
		Field<Agent, 0> cell_id;
		Field<Agent, 1> action;
		Field<Agent, 2> is_alive;
	
		__device__ Agent(int cid);
		__device__ int cell_id();
		__device__ int num_alive_neighbors();
		void _do(void (*pf)(int, int, bool));
};

class Alive : public Agent {
	public:
		declare_field_types(Alive, bool)

		using BaseClass = Agent;

		Field<Alive, 0> is_new;
	
		__device__ Alive(int cid);
		__device__ void prepare();
		__device__ void update();
		__device__ void create_candidates();
		__device__ void maybe_create_candidate(int x, int y);
		void _do(void (*pf)(bool));
};

class Candidate : public Agent {
	public:
		declare_field_types(Candidate)

		using BaseClass = Agent;

	
	
		__device__ Candidate(int cid);
		__device__ void prepare();
		__device__ void update();
		void _do(void (*pf)());
};

extern "C" int Cell_do_all(void (*pf)(Agent*));

extern "C" int Agent_do_all(void (*pf)(int, int, bool));

extern "C" int Candidate_do_all(void (*pf)());

extern "C" int Alive_do_all(void (*pf)(bool));

extern "C" int parallel_new_Cell(int object_num);
extern "C" int parallel_new_Agent(int object_num);
extern "C" int parallel_new_Candidate(int object_num);
extern "C" int parallel_new_Alive(int object_num);
extern "C" int AllocatorInitialize();

extern "C" int AllocatorUninitialize();

#endif