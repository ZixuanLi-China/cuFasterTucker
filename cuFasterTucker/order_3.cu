#include <stdio.h>
#include "kernel.h"

__global__ void Update_Parameter_A_SGD_Order_3(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device, type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		const type_of_data learn_rate_a, const type_of_data lambda_a,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	type_of_data intermediate_variables_shared_0[core_kernel_size];
	type_of_data intermediate_variables_shared_1[core_kernel_size];
	type_of_data parameter_b_shared[core_kernel_size];

#pragma unroll
	for (int kernel_index = 0; kernel_index < core_kernel_size;
			kernel_index++) {
		parameter_b_shared[kernel_index] = parameter_b[(update_order + 2)
				% order][kernel_index * core_dimen + lane_id];
	}
	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

#pragma unroll
		for (int kernel_index = 0; kernel_index < core_kernel_size;
				kernel_index++) {
			intermediate_variables_shared_0[kernel_index] =
					intermediate_variables[update_order][index_0 + kernel_index];
		}

		for (int order_index_1 = start_0; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

#pragma unroll
			for (int kernel_index = 0; kernel_index < core_kernel_size;
					kernel_index++) {

				intermediate_variables_shared_1[kernel_index] =
						intermediate_variables[(update_order + 1) % order][index_1
								+ kernel_index];
			}

			type_of_data gs = 0.0;

#pragma unroll
			for (int kernel_index = 0; kernel_index < core_kernel_size;
					kernel_index++) {
				type_of_data gs_temp = parameter_b_shared[kernel_index];
				gs_temp *= intermediate_variables_shared_0[kernel_index];
				gs_temp *= intermediate_variables_shared_1[kernel_index];
				gs += gs_temp;
			}

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_dimen * order_2;

				type_of_data p_a_gs =
						parameter_a[(update_order + 2) % order][index_2
								+ lane_id] * gs;

				if (core_dimen == 4) {
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
					p_a_gs = __shfl_sync(mask, p_a_gs, 0, 4);
				} else if (core_dimen == 8) {
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
					p_a_gs = __shfl_sync(mask, p_a_gs, 0, 8);
				} else if (core_dimen == 16) {
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 8);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
					p_a_gs = __shfl_sync(mask, p_a_gs, 0, 16);
				} else if (core_dimen == 32) {
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 16);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 8);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
					p_a_gs = __shfl_sync(mask, p_a_gs, 0);
				}

				p_a_gs -= value_train_device[update_order][order_index_2];

				parameter_a[(update_order + 2) % order][index_2 + lane_id] -=
						learn_rate_a
								* (p_a_gs * gs
										+ lambda_a
												* parameter_a[(update_order + 2)
														% order][index_2
														+ lane_id]);
			}
		}
	}
}

__global__ void Update_Parameter_B_CSF_Gradient_Order_3(
		const int order, const int update_order, const int core_kernel,
		const int core_dimen, int nnz, int **ptr_train_len_device,
		int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device,
		type_of_data **value_train_device, type_of_data **parameter_a,
		type_of_data **parameter_b, type_of_data *b_sum,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	type_of_data intermediate_variables_shared_0[core_kernel_size];
	type_of_data intermediate_variables_shared_1[core_kernel_size];
	type_of_data ho_shared[core_kernel_size];
	type_of_data parameter_b_shared[core_kernel_size];

	type_of_data b_gard_temp[core_kernel_size];

#pragma unroll
	for (int kernel_index = 0; kernel_index < core_kernel_size;
			kernel_index++) {
		parameter_b_shared[kernel_index] = parameter_b[(update_order + 2)
				% order][kernel_index * core_dimen + lane_id];
	}

#pragma unroll
	for (int core_kernel_index = 0; core_kernel_index < core_kernel_size;
			core_kernel_index++) {
		b_gard_temp[core_kernel_index] = 0.0;
	}
	__syncthreads();

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

#pragma unroll
		for (int kernel_index = 0; kernel_index < core_kernel_size;
				kernel_index++) {
			intermediate_variables_shared_0[kernel_index] =
					intermediate_variables[update_order][index_0 + kernel_index];
		}

		for (int order_index_1 = start_0; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

#pragma unroll
			for (int kernel_index = 0; kernel_index < core_kernel_size;
					kernel_index++) {
				intermediate_variables_shared_1[kernel_index] =
						intermediate_variables[(update_order + 1) % order][index_1
								+ kernel_index];
			}

			type_of_data gs = 0.0;

#pragma unroll
			for (int kernel_index = 0; kernel_index < core_kernel_size;
					kernel_index++) {

				ho_shared[kernel_index] = 1.0f;
				ho_shared[kernel_index] *=
						intermediate_variables_shared_0[kernel_index];
				ho_shared[kernel_index] *=
						intermediate_variables_shared_1[kernel_index];
				type_of_data gs_temp = parameter_b_shared[kernel_index];
				gs_temp *= ho_shared[kernel_index];
				gs += gs_temp;
			}

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_dimen * order_2;

				type_of_data parameter_a_temp = parameter_a[(update_order + 2)
						% order][index_2 + lane_id];

				type_of_data p_a_gs = parameter_a_temp * gs;

				if (core_dimen == 4) {
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
					p_a_gs = __shfl_sync(mask, p_a_gs, 0, 4);
				} else if (core_dimen == 8) {
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
					p_a_gs = __shfl_sync(mask, p_a_gs, 0, 8);
				} else if (core_dimen == 16) {
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 8);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
					p_a_gs = __shfl_sync(mask, p_a_gs, 0, 16);
				} else if (core_dimen == 32) {
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 16);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 8);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
					p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
					p_a_gs = __shfl_sync(mask, p_a_gs, 0);
				}

				p_a_gs -= value_train_device[update_order][order_index_2];

#pragma unroll
				for (int core_kernel_index = 0;
						core_kernel_index < core_kernel_size;
						core_kernel_index++) {
					b_gard_temp[core_kernel_index] += p_a_gs * parameter_a_temp
							* ho_shared[core_kernel_index];
				}
			}
		}
	}
#pragma unroll
	for (int core_kernel_index = 0; core_kernel_index < core_kernel_size;
			core_kernel_index++) {
		atomicAdd(
				&b_sum[(worker_id % sum_size) * core_kernel * core_dimen
						+ core_kernel_index * core_dimen + lane_id],
				b_gard_temp[core_kernel_index]);
	}
}

void Update_Parameter_A_3(const int order, int *dimen, const int core_kernel,
		const int core_dimen, int nnz_train, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a_device,
		type_of_data **parameter_b_device, const type_of_data learn_rate_a,
		const type_of_data lambda_a, type_of_data **intermediate_variables) {

	for (int update_order = 0; update_order < order; update_order++) {

		Update_Parameter_A_SGD_Order_3 <<<grid_size, block_size
		>>>(order, update_order, core_kernel, core_dimen, nnz_train,
				idx_train_len_device, ptr_train_device, idx_train_device,
				value_train_device, parameter_a_device, parameter_b_device,
				learn_rate_a, lambda_a, intermediate_variables);
		cudaDeviceSynchronize();

		int fact_order = (update_order + order - 1) % order;
		Calculate_Intermediate_Variables
				<<<dimen[fact_order] / block_size * core_dimen, block_size>>>(order,
				core_kernel, core_dimen, parameter_a_device, parameter_b_device,
				fact_order, dimen[fact_order], intermediate_variables);
		cudaDeviceSynchronize();
	}
}

void Update_Parameter_B_Batch_3(const int order, int *dimen,
		const int core_kernel, const int core_dimen, int nnz,
		int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		type_of_data learn_rate_b, type_of_data lambda_b,
		type_of_data **intermediate_variables) {

	type_of_data *b_sum;
	type_of_data *b_grad;

	cudaMalloc((void**) &b_sum,
	sum_size * core_kernel * core_dimen * sizeof(type_of_data));
	cudaMalloc((void**) &b_grad,
			core_kernel * core_dimen * sizeof(type_of_data));

	for (int update_order = 0; update_order < order; update_order++) {

		cudaMemset(b_sum, 0,
		sum_size * core_kernel * core_dimen * sizeof(type_of_data));
		cudaMemset(b_grad, 0, core_kernel * core_dimen * sizeof(type_of_data));

		Update_Parameter_B_CSF_Gradient_Order_3
				<<<grid_size,block_size>>>(order, update_order, core_kernel,
				core_dimen, nnz, ptr_train_len_device, idx_train_len_device,
				ptr_train_device, idx_train_device, value_train_device,
				parameter_a, parameter_b, b_sum, intermediate_variables);

		cudaDeviceSynchronize();

		Parameter_B_Gradient_Sum<<<
		core_kernel / (block_size / core_dimen) + 1, block_size>>>(
				core_kernel, core_dimen, nnz, b_sum, b_grad);
		cudaDeviceSynchronize();

		int fact_order = (update_order + order - 1) % order;

		Update_Parameter_B<<< core_kernel / (block_size / core_dimen) + 1,
		block_size>>>(fact_order, core_kernel, core_dimen, parameter_b, b_grad,
				learn_rate_b, lambda_b);
		cudaDeviceSynchronize();

		Calculate_Intermediate_Variables
				<<<dimen[fact_order] / block_size * core_dimen, block_size>>>(order,
				core_kernel, core_dimen, parameter_a, parameter_b, fact_order,
				dimen[fact_order], intermediate_variables);
		cudaDeviceSynchronize();
	}

	cudaFree(b_sum);
	cudaFree(b_grad);
}
