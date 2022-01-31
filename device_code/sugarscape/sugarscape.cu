#include "sugarscape.h"

AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;
__device__ Cell* cells[kSize * kSize];

__device__ SanajehBaseClass::SanajehBaseClass() {

}

__device__ Cell::Cell(int cell_id) {
	this->agent_ref = nullptr;
	this->sugar_ = 0;
	this->sugar_capacity_ = 3500;
	this->cell_id_ = cell_id;
	curand_init(kSeed, cell_id, 0, &random_state_);
	cells[cell_id] = this;
	int max_grow_rate = 50;
	float r = this->random_float();
	if (r <= 0.02) {
		this->grow_rate_ = max_grow_rate;
	} else 	if (r <= 0.04) {
		this->grow_rate_ = 0.5 * max_grow_rate;
	} else 	if (r <= 0.08) {
		this->grow_rate_ = 0.25 * max_grow_rate;
	} else {
		this->grow_rate_ = 0;
	}
}
__device__ void Cell::Cell__init(int cell_id) {
	this->agent_ref = nullptr;
	this->sugar_ = 0;
	this->sugar_capacity_ = 3500;
	this->cell_id_ = cell_id;
	curand_init(kSeed, cell_id, 0, &random_state_);
	cells[cell_id] = this;
	int max_grow_rate = 50;
	float r = this->random_float();
	if (r <= 0.02) {
		this->grow_rate_ = max_grow_rate;
	} else 	if (r <= 0.04) {
		this->grow_rate_ = 0.5 * max_grow_rate;
	} else 	if (r <= 0.08) {
		this->grow_rate_ = 0.25 * max_grow_rate;
	} else {
		this->grow_rate_ = 0;
	}
}

__device__ void Cell::Setup() {
	float r = this->random_float();
	int c_vision = (kMaxVision / 2) + this->random_int(0, kMaxVision / 2);
	int c_max_age = ((kMaxAge * 2) / 3) + this->random_int(0, kMaxAge / 3);
	int c_endowment = (kMaxEndowment / 4) + this->random_int(0, (kMaxEndowment * 3) / 4);
	int c_metabolism = (kMaxMetabolism / 3) + this->random_int(0, (kMaxMetabolism * 2) / 3);
	int c_max_children = this->random_int(2, kMaxChildren);
	Agent* agent = nullptr;
	if (r < kProbMale) {
		agent = new(device_allocator) Male(this, c_vision, 0, c_max_age, c_endowment, c_metabolism);
	} else 	if (r < kProbMale + kProbFemale) {
		agent = new(device_allocator) Female(this, c_vision, 0, c_max_age, c_endowment, c_metabolism, c_max_children);
	} else {
	
	}
	if (agent != nullptr) {
		this->enter(agent);
	}
}

__device__ void Cell::prepare_diffuse() {
	this->sugar_diffusion_ = kSugarDiffusionRate * this->sugar_;
	int max_diff = kMaxSugarDiffusion;
	if (this->sugar_diffusion_ > max_diff) {
		this->sugar_diffusion_ = max_diff;
	}
	this->sugar_ -= this->sugar_diffusion_;
}

__device__ void Cell::update_diffuse() {
	int new_sugar = 0;
	int self_x = this->cell_id_ % kSize;
	int self_y = this->cell_id_ / kSize;
	for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
		for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
			int nx = self_x + dx;
			int ny = self_y + dy;
			if ((dx != 0 || dy != 0) && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
				int n_id = nx + (ny * kSize);
				Cell* n_cell = cells[n_id];
				new_sugar += 0.125 * n_cell->sugar_diffusion_;
			}
		}
	}
	this->sugar_ += new_sugar;
	if (this->sugar_ > this->sugar_capacity_) {
		this->sugar_ = this->sugar_capacity_;
	}
}

__device__ void Cell::decide_permission() {
	Agent* selected_agent = nullptr;
	int turn = 0;
	int self_x = this->cell_id_ % kSize;
	int self_y = this->cell_id_ / kSize;
	for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
		for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
			int nx = self_x + dx;
			int ny = self_y + dy;
			if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
				int n_id = nx + (ny * kSize);
				Cell* n_cell = cells[n_id];
				Agent* n_agent = n_cell->agent();
				if (n_agent != nullptr && n_agent->cell_request() == this) {
					turn += 1;
					if (this->random_float() <= 1.0 / turn) {
						selected_agent = n_agent;
					} else {
										assert(turn > 1);
					}
				}
			}
		}
	}
	assert(turn == 0 == selected_agent == nullptr);
	if (selected_agent != nullptr) {
		selected_agent->give_permission();
	}
}

__device__ bool Cell::is_free() {
	return this->agent_ref == nullptr;
}

__device__ void Cell::enter(Agent* agent) {
	assert(this->agent_ref == nullptr);
	assert(agent != nullptr);
	this->agent_ref = agent;
	if (this->agent_ref->cast<Male>() != nullptr) {
		this->agent_type = 1;
	} else 	if (this->agent_ref->cast<Female>() != nullptr) {
		this->agent_type = 2;
	} else {
		this->agent_type = 0;
	}
}

__device__ void Cell::leave() {
	assert(this->agent_ref != nullptr);
	this->agent_ref = nullptr;
	this->agent_type = 0;
}

__device__ int Cell::sugar() {
	return this->sugar_;
}

__device__ void Cell::take_sugar(int amount) {
	this->sugar_ -= amount;
}

__device__ void Cell::grow_sugar() {
	this->sugar_ += min(this->sugar_capacity_ - this->sugar_, this->grow_rate_);
}

__device__ float Cell::random_float() {
	return (curand(&random_state_) % 100) * 0.01;
}

__device__ int Cell::random_int(int a, int b) {
	return a + (curand(&random_state_) % ((b - a) + 1));
}

__device__ int Cell::cell_id() {
	return this->cell_id_;
}

__device__ Agent* Cell::agent() {
	return this->agent_ref;
}

__device__ void Cell::add_to_draw_array() {

}

__device__ void Cell::requestTicket() {
	if (this->atomic_request < 200) {
		atomicAdd(&this->atomic_request,1);
	}
}

__device__ Agent::Agent(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism) {
	this->cell_ref = cell;
	this->cell_request_ref = nullptr;
	this->vision_ = vision;
	this->age_ = age;
	this->max_age_ = max_age;
	this->sugar_ = endowment;
	this->endowment_ = endowment;
	this->metabolism_ = metabolism;
	this->permission_ = false;
	assert(cell != nullptr);
	int __auto_v0 = cell->random_int(0, kSize);
	curand_init(kSeed, __auto_v0, 0, &random_state_);
}
__device__ void Agent::Agent__init(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism) {
	this->cell_ref = cell;
	this->cell_request_ref = nullptr;
	this->vision_ = vision;
	this->age_ = age;
	this->max_age_ = max_age;
	this->sugar_ = endowment;
	this->endowment_ = endowment;
	this->metabolism_ = metabolism;
	this->permission_ = false;
	assert(cell != nullptr);
	int __auto_v0 = cell->random_int(0, kSize);
	curand_init(kSeed, __auto_v0, 0, &random_state_);
}

__device__ void Agent::prepare_move() {
	assert(this->cell_ref != nullptr);
	this->cell_ref->requestTicket();
	int turn = 0;
	Cell* target_cell = nullptr;
	int target_sugar = 0;
	int self_x = this->cell_ref->cell_id() % kSize;
	int self_y = this->cell_ref->cell_id() / kSize;
	for (int dx = -this->vision_; dx < this->vision_ + 1; ++dx) {
		for (int dy = -this->vision_; dy < this->vision_ + 1; ++dy) {
			int nx = self_x + dx;
			int ny = self_y + dy;
			if ((dx != 0 || dy != 0) && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
				int n_id = nx + (ny * kSize);
				Cell* n_cell = cells[n_id];
				assert(n_cell != nullptr);
				if (n_cell->is_free()) {
					if (n_cell->sugar() > target_sugar) {
						target_cell = n_cell;
						target_sugar = n_cell->sugar();
						turn = 1;
					} else 					if (n_cell->sugar() == target_sugar) {
						turn += 1;
						if (this->random_float() <= 1.0 / turn) {
							target_cell = n_cell;
						}
					}
				}
			}
		}
	}
	this->cell_request_ref = target_cell;
}

__device__ void Agent::update_move() {
	if (this->permission_ == true) {
		assert(this->cell_request_ref != nullptr);
		assert(this->cell_request_ref->is_free());
		this->cell_ref->leave();
		this->cell_request_ref->enter(this);
		this->cell_ref = this->cell_request_ref;
	}
	this->harvest_sugar();
	this->cell_request_ref = nullptr;
	this->permission_ = false;
}

__device__ void Agent::give_permission() {
	this->permission_ = true;
}

__device__ void Agent::age_and_metabolize() {
	bool dead = false;
	this->age_ += 1;
	dead = this->age_ > this->max_age_;
	this->sugar_ -= this->metabolism_;
	dead = dead || this->sugar_ <= 0;
	if (dead) {
		this->cell_ref->leave();
		destroy(device_allocator, this);
	}
}

__device__ void Agent::harvest_sugar() {
	int amount = this->cell_ref->sugar();
	this->cell_ref->take_sugar(amount);
	this->sugar_ += amount;
}

__device__ bool Agent::ready_to_mate() {
	return this->sugar_ >= (this->endowment_ * 2) / 3 && this->age_ >= kMinMatingAge;
}

__device__ Cell* Agent::cell_request() {
	return this->cell_request_ref;
}

__device__ int Agent::sugar() {
	return this->sugar_;
}

__device__ int Agent::endowment() {
	return this->endowment_;
}

__device__ int Agent::vision() {
	return this->vision_;
}

__device__ int Agent::max_age() {
	return this->max_age_;
}

__device__ int Agent::metabolism() {
	return this->metabolism_;
}

__device__ void Agent::take_sugar(int amount) {
	this->sugar_ = +amount;
}

__device__ float Agent::random_float() {
	return (curand(&random_state_) % 100) * 0.01;
}

__device__ Male::Male(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism) {
this->Agent::Agent__init(cell, vision, age, max_age, endowment, metabolism);
	this->proposal_accepted_ = false;
	this->female_request_ref = nullptr;
}
__device__ void Male::Male__init(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism) {
this->Agent::Agent__init(cell, vision, age, max_age, endowment, metabolism);
	this->proposal_accepted_ = false;
	this->female_request_ref = nullptr;
}

__device__ Female* Male::female_request() {
	return this->female_request_ref;
}

__device__ void Male::accept_proposal() {
	this->proposal_accepted_ = true;
}

__device__ void Male::propose() {
	if (this->ready_to_mate()) {
		Female* target_agent = nullptr;
		int target_sugar = -1;
		int self_x = this->cell_ref->cell_id() % kSize;
		int self_y = this->cell_ref->cell_id() / kSize;
		for (int dx = -this->vision_; dx < this->vision_ + 1; ++dx) {
			for (int dy = -this->vision_; dy < this->vision_ + 1; ++dy) {
				int nx = self_x + dx;
				int ny = self_y + dy;
				if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
					int n_id = nx + (ny * kSize);
					Cell* n_cell = cells[n_id];
					Agent* __auto_v0 = n_cell->agent();
					Female* n_female = __auto_v0->cast<Female>();
					if (n_female->cast<Female>() != nullptr && n_female->ready_to_mate()) {
						if (n_female->sugar() > target_sugar) {
							target_agent = n_female;
							target_sugar = n_female->sugar();
						}
					}
				}
			}
		}
		assert(target_sugar == -1 == target_agent == nullptr);
		this->female_request_ref = target_agent;
	}
}

__device__ void Male::propose_offspring_target() {
	if (this->proposal_accepted_) {
		assert(this->female_request_ref != nullptr);
		Cell* target_cell = nullptr;
		int turn = 0;
		int self_x = this->cell_ref->cell_id() % kSize;
		int self_y = this->cell_ref->cell_id() / kSize;
		for (int dx = -this->vision_; dx < this->vision_ + 1; ++dx) {
			for (int dy = -this->vision_; dy < this->vision_ + 1; ++dy) {
				int nx = self_x + dx;
				int ny = self_y + dy;
				if ((dx != 0 || dy != 0) && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
					int n_id = nx + (ny * kSize);
					Cell* n_cell = cells[n_id];
					if (n_cell->is_free()) {
						turn += 1;
						if (this->random_float() <= 1 / turn) {
							target_cell = n_cell;
						}
					}
				}
			}
		}
		assert(turn == 0 == target_cell == nullptr);
		this->cell_request_ref = target_cell;
	}
}

__device__ void Male::mate() {
	if (this->proposal_accepted_ && this->permission_) {
		assert(this->female_request_ref != nullptr);
		assert(this->cell_request_ref != nullptr);
		this->female_request_ref->increment_num_children();
		int c_endowment = (this->endowment_ + this->female_request_ref->endowment()) / 2;
		this->sugar_ -= this->endowment_ / 2;
		this->female_request_ref->take_sugar(this->female_request_ref->endowment() / 2);
		int c_vision = (this->vision_ + this->female_request_ref->vision()) / 2;
		int c_max_age = (this->max_age_ + this->female_request_ref->max_age()) / 2;
		int c_metabolism = (this->metabolism_ + this->female_request_ref->metabolism()) / 2;
		Agent* child = nullptr;
		if (this->random_float() <= 0.5) {
			child = new(device_allocator) Male(this->cell_request_ref, c_vision, 0, c_max_age, c_endowment, c_metabolism);
		} else {
				int __auto_v0 = this->female_request_ref->max_children();
				child = new(device_allocator) Female(this->cell_request_ref, c_vision, 0, c_max_age, c_endowment, c_metabolism, __auto_v0);
		}
		assert(this->cell_request_ref != nullptr);
		assert(child != nullptr);
		assert(this->cell_request_ref->is_free());
		this->cell_request_ref->enter(child);
	}
	this->permission_ = false;
	this->proposal_accepted_ = false;
	this->female_request_ref = nullptr;
	this->cell_request_ref = nullptr;
}

__device__ Female::Female(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism, int max_children) {
this->Agent::Agent__init(cell, vision, age, max_age, endowment, metabolism);
	this->num_children_ = 0;
	this->max_children_ = max_children;
}
__device__ void Female::Female__init(Cell* cell, int vision, int age, int max_age, int endowment, int metabolism, int max_children) {
this->Agent::Agent__init(cell, vision, age, max_age, endowment, metabolism);
	this->num_children_ = 0;
	this->max_children_ = max_children;
}

__device__ void Female::decide_proposal() {
	if (this->num_children_ < this->max_children_) {
		Male* selected_agent = nullptr;
		int selected_sugar = -1;
		int self_x = this->cell_ref->cell_id() % kSize;
		int self_y = this->cell_ref->cell_id() / kSize;
		for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
			for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
				int nx = self_x + dx;
				int ny = self_y + dy;
				if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
					int n_id = nx + (ny * kSize);
					Cell* n_cell = cells[n_id];
					Agent* __auto_v0 = n_cell->agent();
					Male* n_male = __auto_v0->cast<Male>();
					if (n_male->cast<Male>() != nullptr) {
						if (n_male->female_request() == this && n_male->sugar() > selected_sugar) {
							selected_agent = n_male;
							selected_sugar = n_male->sugar();
						}
					}
				}
			}
		}
		assert(selected_sugar == -1 == selected_agent == nullptr);
		if (selected_agent != nullptr) {
			selected_agent->accept_proposal();
		}
	}
}

__device__ void Female::increment_num_children() {
	this->num_children_ += 1;
}

__device__ int Female::max_children() {
	return this->max_children_;
}

void Cell::_do(void (*pf)(int, int, int, int, int, int, int, int, int)){
	pf(0, 0, this->sugar_diffusion_, this->sugar_, this->sugar_capacity_, this->grow_rate_, this->cell_id_, this->agent_type, this->atomic_request);
}

extern "C" int Cell_do_all(void (*pf)(int, int, int, int, int, int, int, int, int)){
	allocator_handle->template device_do<Cell>(&Cell::_do, pf);
 	return 0;
}

extern "C" int Cell_Cell_Setup(){
	allocator_handle->parallel_do<Cell, &Cell::Setup>();
	return 0;
}

extern "C" int Cell_Cell_grow_sugar(){
	allocator_handle->parallel_do<Cell, &Cell::grow_sugar>();
	return 0;
}

extern "C" int Cell_Cell_prepare_diffuse(){
	allocator_handle->parallel_do<Cell, &Cell::prepare_diffuse>();
	return 0;
}

extern "C" int Cell_Cell_update_diffuse(){
	allocator_handle->parallel_do<Cell, &Cell::update_diffuse>();
	return 0;
}

extern "C" int Agent_Agent_age_and_metabolize(){
	allocator_handle->parallel_do<Agent, &Agent::age_and_metabolize>();
	return 0;
}

extern "C" int Agent_Agent_prepare_move(){
	allocator_handle->parallel_do<Agent, &Agent::prepare_move>();
	return 0;
}

extern "C" int Cell_Cell_decide_permission(){
	allocator_handle->parallel_do<Cell, &Cell::decide_permission>();
	return 0;
}

extern "C" int Agent_Agent_update_move(){
	allocator_handle->parallel_do<Agent, &Agent::update_move>();
	return 0;
}

extern "C" int Male_Male_propose(){
	allocator_handle->parallel_do<Male, &Male::propose>();
	return 0;
}

extern "C" int Female_Female_decide_proposal(){
	allocator_handle->parallel_do<Female, &Female::decide_proposal>();
	return 0;
}

extern "C" int Male_Male_propose_offspring_target(){
	allocator_handle->parallel_do<Male, &Male::propose_offspring_target>();
	return 0;
}

extern "C" int Male_Male_mate(){
	allocator_handle->parallel_do<Male, &Male::mate>();
	return 0;
}

extern "C" int parallel_new_Cell(int object_num){
	allocator_handle->parallel_new<Cell>(object_num);
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