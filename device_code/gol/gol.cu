#include "gol.h"

AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;
__device__ Cell** cells;
Cell** host_cells;

__device__ Cell::Cell(int idx) {
	this->agent_ = nullptr;
	cells[idx] = this;
}

__device__ Agent* Cell::agent() {
	return this->agent_;
}

__device__ bool Cell::is_empty() {
	return this->agent_ == nullptr;
}

__device__ Agent::Agent(int cid) {
	this->cell_id = cid;
	this->action = 0;
	this->is_alive = false;
}

__device__ int Agent::cell_id() {
	return this->cell_id;
}

__device__ int Agent::num_alive_neighbors() {
	int cell_x = this->cell_id % SIZE_X;
	int cell_y = this->cell_id / SIZE_Y;
	int result = 0;
	int dx = -1;
	int dy = -1;
	while (dx < 2) {
		while (dy < 2) {
			int nx = cell_x + dx;
			int ny = cell_y + dy;
			if (-1 < nx < SIZE_X && -1 < ny < SIZE_Y) {
				if (cells[(ny * SIZE_X) + nx]->agent()->is_alive) {
					result += 1;
				}
			}
			dy += 1;
		}
		dx += 1;
	}
	return result;
}

__device__ Alive::Alive(int cid) : Agent(cid) {
	this->is_new = true;
}

__device__ void Alive::prepare() {
	this->is_new = false;
	int alive_neighbors = this->num_alive_neighbors() - 1;
	if (alive_neighbors < 2 || alive_neighbors > 3) {
		this->action = 1;
	}
}

__device__ void Alive::update() {
	int cid = this->cell_id;
	if (this->is_new) {
		this->create_candidates();
	} else 	if (this->action == 1) {
		cells[cid]->agent_ = new(device_allocator) Candidate(cid);
		destroy(device_allocator, this);
	}
}

__device__ void Alive::create_candidates() {
	int cell_x = this->cell_id % SIZE_X;
	int cell_y = this->cell_id / SIZE_Y;
	int dx = -1;
	int dy = -1;
	while (dx < 2) {
		while (dy < 2) {
			int nx = cell_x + dx;
			int ny = cell_y + dy;
			if (-1 < nx < SIZE_X && -1 < ny < SIZE_Y) {
				if (cells[(ny * SIZE_X) + nx]->is_empty()) {
					this->maybe_create_candidate(nx, ny);
				}
			}
			dy += 1;
		}
		dx += 1;
	}
}

__device__ void Alive::maybe_create_candidate(int x, int y) {
	int dx = -1;
	int dy = -1;
	while (dx < 2) {
		while (dy < 2) {
			int nx = x + dx;
			int ny = y + dy;
			if (-1 < nx < SIZE_X && -1 < ny < SIZE_Y) {
				if (cells[(ny * SIZE_X) + nx]->agent()->is_alive) {
					Alive* alive = cells[(ny * SIZE_X) + nx]->agent();
					if (alive->is_new) {
						if (alive == this) {
							cells[(y * SIZE_X) + x]->agent_ = new(device_allocator) Candidate((y * SIZE_X) + x);
						}
						return;
					}
				}
			}
			dy += 1;
		}
		dx += 1;
	}
}

__device__ Candidate::Candidate(int cid) : Agent(cid) {

}

__device__ void Candidate::prepare() {
	int alive_neighbors = this->num_alive_neighbors();
	if (alive_neighbors == 3) {
		this->action = 2;
	} else 	if (alive_neighbors == 0) {
		this->action = 1;
	}
}

__device__ void Candidate::update() {
	int cid = this->cell_id;
	if (this->action == 2) {
		cells[cid]->agent_ = new(device_allocator) Alive(cid);
		cells[cid]->agent_->is_alive = true;
		destroy(device_allocator, this);
	}
}

void Cell::_do(void (*pf)(Agent*)){
	pf(this->agent_);
}

extern "C" int Cell_do_all(void (*pf)(Agent*)){
	allocator_handle->template device_do<Cell>(&Cell::_do, pf);
 	return 0;
}


void Agent::_do(void (*pf)(int, int, bool)){
	pf(this->cell_id, this->action, this->is_alive);
}

extern "C" int Agent_do_all(void (*pf)(int, int, bool)){
	allocator_handle->template device_do<Agent>(&Agent::_do, pf);
 	return 0;
}


void Candidate::_do(void (*pf)()){
	pf();
}

extern "C" int Candidate_do_all(void (*pf)()){
	allocator_handle->template device_do<Candidate>(&Candidate::_do, pf);
 	return 0;
}


void Alive::_do(void (*pf)(bool)){
	pf(this->is_new);
}

extern "C" int Alive_do_all(void (*pf)(bool)){
	allocator_handle->template device_do<Alive>(&Alive::_do, pf);
 	return 0;
}



extern "C" int parallel_new_Cell(int object_num){
	allocator_handle->parallel_new<Cell>(object_num);
	return 0;
}

extern "C" int parallel_new_Agent(int object_num){
	allocator_handle->parallel_new<Agent>(object_num);
	return 0;
}

extern "C" int parallel_new_Candidate(int object_num){
	allocator_handle->parallel_new<Candidate>(object_num);
	return 0;
}

extern "C" int parallel_new_Alive(int object_num){
	allocator_handle->parallel_new<Alive>(object_num);
	return 0;
}

extern "C" int AllocatorInitialize(){
	allocator_handle = new AllocatorHandle<AllocatorT>(/* unified_memory= */ true);
	AllocatorT* dev_ptr = allocator_handle->device_pointer();
	cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0, cudaMemcpyHostToDevice);
	Cell** host_cells;
	cudaMalloc(&host_cells, sizeof(Cell*)*1000);
	cudaMemcpyToSymbol(cells, &host_cells, sizeof(Cell**), 0, cudaMemcpyHostToDevice);
	return 0;
}

extern "C" int AllocatorUninitialize(){
	cudaFree(host_cells);
	return 0;
}