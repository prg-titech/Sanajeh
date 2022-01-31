#ifndef SANAJEH_DEVICE_CODE_H
#define SANAJEH_DEVICE_CODE_H
#define KNUMOBJECTS 64*64*64*64

#include <curand_kernel.h>
#include "dynasoar.h"

class SanajehBaseClass;
class Cell;
class Agent;
class Male;
class Female;


using AllocatorT = SoaAllocator<KNUMOBJECTS, SanajehBaseClass, Cell, Agent, Male, Female>;

static const int kSize = 100;
static const int kSeed = 42;
static const float kProbMale = 0.12;
static const float kProbFemale = 0.15;
static const int kMaxVision = 2;
static const int kMaxAge = 100;
static const int kMaxEndowment = 200;
static const int kMaxMetabolism = 80;
static const int kMaxSugarDiffusion = 60;
static const float kSugarDiffusionRate = 0.125;
static const int kMinMatingAge = 22;
static const int kMaxChildren = 8;

class SanajehBaseClass : public AllocatorT::Base {
	public:
		declare_field_types(SanajehBaseClass, int)
		Field<SanajehBaseClass, 0> class_type;
	
		__device__ SanajehBaseClass();
		void _do(void (*pf)(int));
};

class Cell : public SanajehBaseClass {
	public:
		declare_field_types(Cell, curandState, Agent*, int, int, int, int, int, int, int)

		using BaseClass = SanajehBaseClass;

		Field<Cell, 0> random_state_;
		Field<Cell, 1> agent_ref;
		Field<Cell, 2> sugar_diffusion_;
		Field<Cell, 3> sugar_;
		Field<Cell, 4> sugar_capacity_;
		Field<Cell, 5> grow_rate_;
		Field<Cell, 6> cell_id_;
		Field<Cell, 7> agent_type;
		Field<Cell, 8> atomic_request;

__device__ Cell() {};
		
		__device__ Cell(int cell_id);
	__device__ void Cell__init(int cell_id);
		__device__ void Setup();
		__device__ void prepare_diffuse();
		__device__ void update_diffuse();
		__device__ void decide_permission();
		__device__ bool is_free();
		__device__ void enter(Agent* agent);
		__device__ void leave();
		__device__ int sugar();
		__device__ void take_sugar(int amount);
		__device__ void grow_sugar();
		__device__ float random_float();
		__device__ int random_int(int a, int b);
		__device__ int cell_id();
		__device__ Agent* agent();
		__device__ void add_to_draw_array();
		__device__ void requestTicket();
		void _do(void (*pf)(int, int, int, int, int, int, int, int, int));
};

class Agent : public SanajehBaseClass {
	public:
		declare_field_types(Agent, curandState, Cell*, Cell*, int, int, int, int, int, int, bool)

		using BaseClass = SanajehBaseClass;

		Field<Agent, 0> random_state_;
		Field<Agent, 1> cell_ref;
		Field<Agent, 2> cell_request_ref;
		Field<Agent, 3> vision_;
		Field<Agent, 4> age_;
		Field<Agent, 5> max_age_;
		Field<Agent, 6> sugar_;
		Field<Agent, 7> metabolism_;
		Field<Agent, 8> endowment_;
		Field<Agent, 9> permission_;

__device__ Agent() {};
		
		static const bool kIsAbstract = true;
		__device__ Agent(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism);
	__device__ void Agent__init(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism);
		__device__ void prepare_move();
		__device__ void update_move();
		__device__ void give_permission();
		__device__ void age_and_metabolize();
		__device__ void harvest_sugar();
		__device__ bool ready_to_mate();
		__device__ Cell* cell_request();
		__device__ int sugar();
		__device__ int endowment();
		__device__ int vision();
		__device__ int max_age();
		__device__ int metabolism();
		__device__ void take_sugar(int amount);
		__device__ float random_float();
		void _do(void (*pf)(int, int, int, int, int, int, int, int, int, bool));
};

class Male : public Agent {
	public:
		declare_field_types(Male, Female*, bool)

		using BaseClass = Agent;

		Field<Male, 0> female_request_ref;
		Field<Male, 1> proposal_accepted_;

__device__ Male() {};
		
		static const bool kIsAbstract = false;
		__device__ Male(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism);
	__device__ void Male__init(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism);
		__device__ Female* female_request();
		__device__ void accept_proposal();
		__device__ void propose();
		__device__ void propose_offspring_target();
		__device__ void mate();
		void _do(void (*pf)(int, bool));
};

class Female : public Agent {
	public:
		declare_field_types(Female, int, int)

		using BaseClass = Agent;

		Field<Female, 0> max_children_;
		Field<Female, 1> num_children_;

__device__ Female() {};
		
		static const bool kIsAbstract = false;
		__device__ Female(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism, int max_children);
	__device__ void Female__init(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism, int max_children);
		__device__ void decide_proposal();
		__device__ void increment_num_children();
		__device__ int max_children();
		void _do(void (*pf)(int, int));
};

extern "C" int Cell_do_all(void (*pf)(int, int, int, int, int, int, int, int, int));
extern "C" int Cell_Cell_Setup();
extern "C" int Cell_Cell_grow_sugar();
extern "C" int Cell_Cell_prepare_diffuse();
extern "C" int Cell_Cell_update_diffuse();
extern "C" int Agent_Agent_age_and_metabolize();
extern "C" int Agent_Agent_prepare_move();
extern "C" int Cell_Cell_decide_permission();
extern "C" int Agent_Agent_update_move();
extern "C" int Male_Male_propose();
extern "C" int Female_Female_decide_proposal();
extern "C" int Male_Male_propose_offspring_target();
extern "C" int Male_Male_mate();
extern "C" int parallel_new_Cell(int object_num);
extern "C" int AllocatorInitialize();

extern "C" int AllocatorUninitialize();

#endif