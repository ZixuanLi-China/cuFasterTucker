#include <cublas_v2.h>
#include "order_3.h"
#include "order_4.h"
#include "order_5.h"
#include "order_6.h"
#include "order_7.h"
#include "order_8.h"
#include "order_9.h"
#include "order_10.h"

__global__ void Calculate_Intermediate_Variables(const int order,
		const int core_kernel, const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int update_order,
		const int update_order_dimen,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int update_order_dimen_index = worker_id;
			update_order_dimen_index < update_order_dimen;
			update_order_dimen_index += workers) {

		for (int core_kernel_index = 0; core_kernel_index < core_kernel;
				core_kernel_index++) {

			type_of_data temp =
					parameter_a[update_order][update_order_dimen_index
							* core_dimen + lane_id]
							* parameter_b[update_order][core_kernel_index
									* core_dimen + lane_id];

			if (core_dimen == 4) {
				temp += __shfl_down_sync(mask, temp, 2);
				temp += __shfl_down_sync(mask, temp, 1);
				temp = __shfl_sync(mask, temp, 0, 4);
			} else if (core_dimen == 8) {
				temp += __shfl_down_sync(mask, temp, 4);
				temp += __shfl_down_sync(mask, temp, 2);
				temp += __shfl_down_sync(mask, temp, 1);
				temp = __shfl_sync(mask, temp, 0, 8);
			} else if (core_dimen == 16) {
				temp += __shfl_down_sync(mask, temp, 8);
				temp += __shfl_down_sync(mask, temp, 4);
				temp += __shfl_down_sync(mask, temp, 2);
				temp += __shfl_down_sync(mask, temp, 1);
				temp = __shfl_sync(mask, temp, 0, 16);
			} else if (core_dimen == 32) {
				temp += __shfl_down_sync(mask, temp, 16);
				temp += __shfl_down_sync(mask, temp, 8);
				temp += __shfl_down_sync(mask, temp, 4);
				temp += __shfl_down_sync(mask, temp, 2);
				temp += __shfl_down_sync(mask, temp, 1);
				temp = __shfl_sync(mask, temp, 0);
			}

			if (lane_id == 0) {
				intermediate_variables[update_order][update_order_dimen_index
						* core_kernel + core_kernel_index] = temp;
			}
		}
	}
}

void Intermediate_Variables_Initialization(const int order, int *dimen,
		const int core_kernel, const int core_dimen,
		type_of_data **parameter_a_device, type_of_data **parameter_b_device,
		type_of_data **intermediate_variables_device) {

	for (int i = 0; i < order; i++) {
		Calculate_Intermediate_Variables
				<<<dimen[i] / block_size * core_dimen, block_size>>>(order,
				core_kernel, core_dimen, parameter_a_device, parameter_b_device,
				i, dimen[i], intermediate_variables_device);
		cudaDeviceSynchronize();
	}
}

__global__ void Update_Parameter_A(const int update_order, const int dimen,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data **a_grad_up,
		type_of_data **a_grad_down) {
	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int a_index = worker_id; a_index < dimen; a_index += workers) {

		if ((a_grad_up[update_order][a_index * core_dimen + lane_id]
				/ a_grad_down[update_order][a_index * core_dimen + lane_id])
				== (a_grad_up[update_order][a_index * core_dimen + lane_id]
						/ a_grad_down[update_order][a_index * core_dimen
								+ lane_id])) {
			parameter_a[update_order][a_index * core_dimen + lane_id] *=
					(a_grad_up[update_order][a_index * core_dimen + lane_id]
							/ a_grad_down[update_order][a_index * core_dimen
									+ lane_id]);
		}

	}
}

void Update_Parameter_A(const int order, int *dimen, const int core_kernel,
		const int core_dimen, int nnz_train, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a_device,
		type_of_data **parameter_b_device, const type_of_data lambda_a,
		type_of_data **intermediate_variables,
		type_of_data **a_grad_up,
		type_of_data **a_grad_down, type_of_data **a_grad_up_host_to_device,
		type_of_data **a_grad_down_host_to_device) {

	if (order == 3) {
		Update_Parameter_A_3(order, dimen, core_kernel, core_dimen, nnz_train,
				idx_train_len_device, ptr_train_device, idx_train_device,
				value_train_device, parameter_a_device, parameter_b_device,
				lambda_a, intermediate_variables, a_grad_up, a_grad_down,
				a_grad_up_host_to_device, a_grad_down_host_to_device);
	} else if (order == 4) {
		Update_Parameter_A_4(order, dimen, core_kernel, core_dimen, nnz_train,
				idx_train_len_device, ptr_train_device, idx_train_device,
				value_train_device, parameter_a_device, parameter_b_device,
				lambda_a, intermediate_variables, a_grad_up, a_grad_down,
				a_grad_up_host_to_device, a_grad_down_host_to_device);
	} else if (order == 5) {
		Update_Parameter_A_5(order, dimen, core_kernel, core_dimen, nnz_train,
				idx_train_len_device, ptr_train_device, idx_train_device,
				value_train_device, parameter_a_device, parameter_b_device,
				lambda_a, intermediate_variables, a_grad_up, a_grad_down,
				a_grad_up_host_to_device, a_grad_down_host_to_device);
	} else if (order == 6) {
		Update_Parameter_A_6(order, dimen, core_kernel, core_dimen, nnz_train,
				idx_train_len_device, ptr_train_device, idx_train_device,
				value_train_device, parameter_a_device, parameter_b_device,
				lambda_a, intermediate_variables, a_grad_up, a_grad_down,
				a_grad_up_host_to_device, a_grad_down_host_to_device);
	} else if (order == 7) {
		Update_Parameter_A_7(order, dimen, core_kernel, core_dimen, nnz_train,
				idx_train_len_device, ptr_train_device, idx_train_device,
				value_train_device, parameter_a_device, parameter_b_device,
				lambda_a, intermediate_variables, a_grad_up, a_grad_down,
				a_grad_up_host_to_device, a_grad_down_host_to_device);
	} else if (order == 8) {
		Update_Parameter_A_8(order, dimen, core_kernel, core_dimen, nnz_train,
				idx_train_len_device, ptr_train_device, idx_train_device,
				value_train_device, parameter_a_device, parameter_b_device,
				lambda_a, intermediate_variables, a_grad_up, a_grad_down,
				a_grad_up_host_to_device, a_grad_down_host_to_device);
	} else if (order == 9) {
		Update_Parameter_A_9(order, dimen, core_kernel, core_dimen, nnz_train,
				idx_train_len_device, ptr_train_device, idx_train_device,
				value_train_device, parameter_a_device, parameter_b_device,
				lambda_a, intermediate_variables, a_grad_up, a_grad_down,
				a_grad_up_host_to_device, a_grad_down_host_to_device);
	} else if (order == 10) {
		Update_Parameter_A_10(order, dimen, core_kernel, core_dimen, nnz_train,
				idx_train_len_device, ptr_train_device, idx_train_device,
				value_train_device, parameter_a_device, parameter_b_device,
				lambda_a, intermediate_variables, a_grad_up, a_grad_down,
				a_grad_up_host_to_device, a_grad_down_host_to_device);
	}

}

__global__ void Parameter_B_Gradient_Sum_Up(const int core_kernel,
		const int core_dimen,
		type_of_data *b_sum, type_of_data *b_grad) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int core_kernel_index = worker_id; core_kernel_index < core_kernel;
			core_kernel_index += workers) {
		for (int sum_size_index = 0; sum_size_index < sum_size;
				sum_size_index++) {
			b_grad[core_kernel_index * core_dimen + lane_id] +=
					b_sum[core_kernel_index * core_dimen + lane_id];
		}
	}
}

__global__ void Parameter_B_Gradient_Sum_Down(const int core_kernel,
		const int core_dimen, const int nnz,
		type_of_data *b_sum, type_of_data *b_grad, type_of_data **parameter_b,
		const type_of_data lambda_b, const int update_order) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int core_kernel_index = worker_id; core_kernel_index < core_kernel;
			core_kernel_index += workers) {
		for (int sum_size_index = 0; sum_size_index < sum_size;
				sum_size_index++) {
			b_grad[core_kernel_index * core_dimen + lane_id] +=
					b_sum[core_kernel_index * core_dimen + lane_id];
		}
		b_grad[core_kernel_index * core_dimen + lane_id] += nnz * lambda_b
				* parameter_b[update_order][core_kernel_index * core_dimen
						+ lane_id];
	}
}

__global__ void Update_Parameter_B(const int update_order,
		const int core_kernel, const int core_dimen, type_of_data **parameter_b,
		type_of_data *b_grad_up, type_of_data *b_grad_down) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int core_kernel_index = worker_id; core_kernel_index < core_kernel;
			core_kernel_index += workers) {

		if ((b_grad_up[core_kernel_index * core_dimen + lane_id]
				/ b_grad_down[core_kernel_index * core_dimen + lane_id])
				== (b_grad_up[core_kernel_index * core_dimen + lane_id]
						/ b_grad_down[core_kernel_index * core_dimen + lane_id])) {
			parameter_b[update_order][core_kernel_index * core_kernel + lane_id] *=
					(b_grad_up[core_kernel_index * core_dimen + lane_id]
							/ b_grad_down[core_kernel_index * core_dimen
									+ lane_id]);
		}
	}
}

void Update_Parameter_B_Batch(const int order, int *dimen,
		const int core_kernel, const int core_dimen, int nnz,
		int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		type_of_data lambda_b,
		type_of_data **intermediate_variables) {

	if (order == 3) {
		Update_Parameter_B_Batch_3(order, dimen, core_kernel, core_dimen, nnz,
				ptr_train_len_device, idx_train_len_device, ptr_train_device,
				idx_train_device, value_train_device, parameter_a, parameter_b,
				lambda_b, intermediate_variables);
	} else if (order == 4) {
		Update_Parameter_B_Batch_4(order, dimen, core_kernel, core_dimen, nnz,
				ptr_train_len_device, idx_train_len_device, ptr_train_device,
				idx_train_device, value_train_device, parameter_a, parameter_b,
				lambda_b, intermediate_variables);
	} else if (order == 5) {
		Update_Parameter_B_Batch_5(order, dimen, core_kernel, core_dimen, nnz,
				ptr_train_len_device, idx_train_len_device, ptr_train_device,
				idx_train_device, value_train_device, parameter_a, parameter_b,
				lambda_b, intermediate_variables);
	} else if (order == 6) {
		Update_Parameter_B_Batch_6(order, dimen, core_kernel, core_dimen, nnz,
				ptr_train_len_device, idx_train_len_device, ptr_train_device,
				idx_train_device, value_train_device, parameter_a, parameter_b,
				lambda_b, intermediate_variables);
	} else if (order == 7) {
		Update_Parameter_B_Batch_7(order, dimen, core_kernel, core_dimen, nnz,
				ptr_train_len_device, idx_train_len_device, ptr_train_device,
				idx_train_device, value_train_device, parameter_a, parameter_b,
				lambda_b, intermediate_variables);
	} else if (order == 8) {
		Update_Parameter_B_Batch_8(order, dimen, core_kernel, core_dimen, nnz,
				ptr_train_len_device, idx_train_len_device, ptr_train_device,
				idx_train_device, value_train_device, parameter_a, parameter_b,
				lambda_b, intermediate_variables);
	} else if (order == 9) {
		Update_Parameter_B_Batch_9(order, dimen, core_kernel, core_dimen, nnz,
				ptr_train_len_device, idx_train_len_device, ptr_train_device,
				idx_train_device, value_train_device, parameter_a, parameter_b,
				lambda_b, intermediate_variables);
	} else if (order == 10) {
		Update_Parameter_B_Batch_10(order, dimen, core_kernel, core_dimen, nnz,
				ptr_train_len_device, idx_train_len_device, ptr_train_device,
				idx_train_device, value_train_device, parameter_a, parameter_b,
				lambda_b, intermediate_variables);
	}
}

__global__ void RMSE_AND_MAE_CSF_3(const int order, const int update_order,
		const int core_kernel, const int core_dimen, int **ptr_train_len_device,
		int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device,
		type_of_data **value_train_device, type_of_data **parameter_a,
		type_of_data **parameter_b, type_of_data *rmse,
		type_of_data *mae) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;
	int parallel = idx_train_len_device[update_order][0];

	for (int order_index_0 = worker_id; order_index_0 < parallel;
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_dimen * order_0;

		for (int order_index_1 = start_0; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_dimen * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int oder_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_dimen * oder_2;
				type_of_data val =
						value_train_device[update_order][order_index_2];

				for (int kernel_index = 0; kernel_index < core_kernel;
						kernel_index++) {

					type_of_data temp_0 =
							parameter_a[0][index_0 + lane_id]
									* parameter_b[0][kernel_index * core_dimen
											+ lane_id];
					type_of_data temp_1 =
							parameter_a[1][index_1 + lane_id]
									* parameter_b[1][kernel_index * core_dimen
											+ lane_id];
					type_of_data temp_2 =
							parameter_a[2][index_2 + lane_id]
									* parameter_b[2][kernel_index * core_dimen
											+ lane_id];

					if (core_dimen == 4) {
						temp_0 += __shfl_down_sync(mask, temp_0, 2);
						temp_0 += __shfl_down_sync(mask, temp_0, 1);
						temp_0 = __shfl_sync(mask, temp_0, 0, 4);
						temp_1 += __shfl_down_sync(mask, temp_1, 2);
						temp_1 += __shfl_down_sync(mask, temp_1, 1);
						temp_1 = __shfl_sync(mask, temp_1, 0, 4);
						temp_2 += __shfl_down_sync(mask, temp_2, 2);
						temp_2 += __shfl_down_sync(mask, temp_2, 1);
						temp_2 = __shfl_sync(mask, temp_2, 0, 4);
					} else if (core_dimen == 8) {
						temp_0 += __shfl_down_sync(mask, temp_0, 4);
						temp_0 += __shfl_down_sync(mask, temp_0, 2);
						temp_0 += __shfl_down_sync(mask, temp_0, 1);
						temp_0 = __shfl_sync(mask, temp_0, 0, 8);
						temp_1 += __shfl_down_sync(mask, temp_1, 4);
						temp_1 += __shfl_down_sync(mask, temp_1, 2);
						temp_1 += __shfl_down_sync(mask, temp_1, 1);
						temp_1 = __shfl_sync(mask, temp_1, 0, 8);
						temp_2 += __shfl_down_sync(mask, temp_2, 4);
						temp_2 += __shfl_down_sync(mask, temp_2, 2);
						temp_2 += __shfl_down_sync(mask, temp_2, 1);
						temp_2 = __shfl_sync(mask, temp_2, 0, 8);
					} else if (core_dimen == 16) {
						temp_0 += __shfl_down_sync(mask, temp_0, 8);
						temp_0 += __shfl_down_sync(mask, temp_0, 4);
						temp_0 += __shfl_down_sync(mask, temp_0, 2);
						temp_0 += __shfl_down_sync(mask, temp_0, 1);
						temp_0 = __shfl_sync(mask, temp_0, 0, 16);
						temp_1 += __shfl_down_sync(mask, temp_1, 8);
						temp_1 += __shfl_down_sync(mask, temp_1, 4);
						temp_1 += __shfl_down_sync(mask, temp_1, 2);
						temp_1 += __shfl_down_sync(mask, temp_1, 1);
						temp_1 = __shfl_sync(mask, temp_1, 0, 16);
						temp_2 += __shfl_down_sync(mask, temp_2, 8);
						temp_2 += __shfl_down_sync(mask, temp_2, 4);
						temp_2 += __shfl_down_sync(mask, temp_2, 2);
						temp_2 += __shfl_down_sync(mask, temp_2, 1);
						temp_2 = __shfl_sync(mask, temp_2, 0, 16);
					} else if (core_dimen == 32) {
						temp_0 += __shfl_down_sync(mask, temp_0, 16);
						temp_0 += __shfl_down_sync(mask, temp_0, 8);
						temp_0 += __shfl_down_sync(mask, temp_0, 4);
						temp_0 += __shfl_down_sync(mask, temp_0, 2);
						temp_0 += __shfl_down_sync(mask, temp_0, 1);
						temp_0 = __shfl_sync(mask, temp_0, 0);
						temp_1 += __shfl_down_sync(mask, temp_1, 16);
						temp_1 += __shfl_down_sync(mask, temp_1, 8);
						temp_1 += __shfl_down_sync(mask, temp_1, 4);
						temp_1 += __shfl_down_sync(mask, temp_1, 2);
						temp_1 += __shfl_down_sync(mask, temp_1, 1);
						temp_1 = __shfl_sync(mask, temp_1, 0);
						temp_2 += __shfl_down_sync(mask, temp_2, 16);
						temp_2 += __shfl_down_sync(mask, temp_2, 8);
						temp_2 += __shfl_down_sync(mask, temp_2, 4);
						temp_2 += __shfl_down_sync(mask, temp_2, 2);
						temp_2 += __shfl_down_sync(mask, temp_2, 1);
						temp_2 = __shfl_sync(mask, temp_2, 0);
					}
					val -= temp_0 * temp_1 * temp_2;
				}

				if (lane_id == 0) {
					atomicAdd(&rmse[order_index_2 % error_size], val * val);
					atomicAdd(&mae[order_index_2 % error_size], abs(val));
				}
			}
		}
	}
}

__global__ void RMSE_AND_MAE_COO(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, const type_of_data *value,
		int **index, type_of_data *rmse,
		type_of_data *mae) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_wid = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_wid;
	int workers = worker * gridDim.x;

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		type_of_data p_a_gs = 0.0;
		type_of_data gs = 0.0;

		for (int core_kernel_index = 0; core_kernel_index < core_kernel;
				core_kernel_index++) {
			type_of_data gs_temp = parameter_b[0][core_kernel_index * core_dimen
					+ lane_id];

			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != 0) {
					type_of_data temp =
							parameter_a[inner_order_index][index[inner_order_index][nnz_index]
									* core_dimen + lane_id]
									* parameter_b[inner_order_index][core_kernel_index
											* core_dimen + lane_id];

					if (core_dimen == 4) {
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 4);
					} else if (core_dimen == 8) {
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 8);
					} else if (core_dimen == 16) {
						temp += __shfl_down_sync(mask, temp, 8);
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 16);
					} else if (core_dimen == 32) {
						temp += __shfl_down_sync(mask, temp, 16);
						temp += __shfl_down_sync(mask, temp, 8);
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0);
					}

					gs_temp *= temp;

				}
			}
			gs += gs_temp;
		}

		p_a_gs = parameter_a[0][index[0][nnz_index] * core_dimen + lane_id]
				* gs;

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

		p_a_gs -= value[nnz_index];

		if (lane_id == 0) {
			atomicAdd(&rmse[nnz_index % error_size], p_a_gs * p_a_gs);
			atomicAdd(&mae[nnz_index % error_size], abs(p_a_gs));
		}

	}

}

void GET_RMSE_AND_MAE_CSF_3(const int order, const int core_kernel,
		const int core_dimen, const int nnz, int **ptr_train_len_device,
		int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		type_of_data *rmse,
		type_of_data *mae) {

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse,
	error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0, error_size * sizeof(type_of_data));

	RMSE_AND_MAE_CSF_3<<<grid_size, block_size>>>(order, 0, core_kernel,
			core_dimen, ptr_train_len_device, idx_train_len_device,
			ptr_train_device, idx_train_device, value_train_device, parameter_a,
			parameter_b, errors_rmse, errors_mae);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));
	type_of_data *mae_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();
	cublasSasum(handle_mae, error_size, errors_mae, 1, mae_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	*mae = (*mae_sum) / nnz;
	cublasDestroy(handle_rmse);
	cublasDestroy(handle_mae);
	cudaFree(errors_rmse);
	cudaFree(errors_mae);
	free(rmse_sum);
	free(mae_sum);

}

void GET_RMSE_AND_MAE_COO(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		type_of_data *value, int **index,
		type_of_data *rmse,
		type_of_data *mae) {

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse,
	error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0, error_size * sizeof(type_of_data));

	RMSE_AND_MAE_COO<<<nnz / block_size + 1, block_size>>>(order, core_kernel,
			core_dimen, parameter_a, parameter_b, nnz, value, index,
			errors_rmse, errors_mae);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));
	type_of_data *mae_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();
	cublasSasum(handle_mae, error_size, errors_mae, 1, mae_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	*mae = (*mae_sum) / nnz;
	cublasDestroy(handle_rmse);
	cublasDestroy(handle_mae);
	cudaFree(errors_rmse);
	cudaFree(errors_mae);
	free(rmse_sum);
	free(mae_sum);

}
