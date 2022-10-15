#include "kernel.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "tools.h"

#define type_of_data float
#define grid_size 1024*2
#define block_size 128
#define warp_size 32
#define sum_size 1024
#define error_size 1024

//#define mask 0xffffffff
#define mask 0x00000000

using namespace std;

__global__ void Calculate_Intermediate_Variables(const int order,
		const int core_kernel, const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int update_order,
		const int update_order_dimen,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
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

			int temp_temp = core;
			while (temp_temp != 1) {
				temp_temp /= 2;
				temp += __shfl_down_sync(mask, temp, temp_temp);
			}

			temp = __shfl_sync(mask, temp, (local_id % local) * core);
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

__global__ void Update_Parameter_A_SGD_Order_3(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device, type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		const type_of_data learn_rate_a, const type_of_data lambda_a,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				type_of_data gs = 0.0;

				for (int kernel_index = 0; kernel_index < core_kernel;
						kernel_index++) {
					type_of_data gs_temp = parameter_b[(update_order + 2)
							% order][kernel_index * core_dimen + lane_id];
					gs_temp *= intermediate_variables[update_order][index_0
							+ kernel_index];
					gs_temp *=
							intermediate_variables[(update_order + 1) % order][index_1
									+ kernel_index];
					gs += gs_temp;
				}

				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_dimen * order_2;

				type_of_data p_a_gs =
						parameter_a[(update_order + 2) % order][index_2
								+ lane_id] * gs;

				int temp = core;
				while (temp != 1) {
					temp /= 2;
					p_a_gs += __shfl_down_sync(mask, p_a_gs, temp);
				}

				p_a_gs = __shfl_sync(mask, p_a_gs, (local_id % local) * core);

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

__global__ void Update_Parameter_A_SGD_Order_4(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device, type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		const type_of_data learn_rate_a, const type_of_data lambda_a,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				type_of_data gs = 0.0;

				for (int kernel_index = 0; kernel_index < core_kernel;
						kernel_index++) {
					type_of_data gs_temp = parameter_b[(update_order + 3)
							% order][kernel_index * core_dimen + lane_id];

					gs_temp *= intermediate_variables[update_order][index_0
							+ kernel_index];
					gs_temp *=
							intermediate_variables[(update_order + 1) % order][index_1
									+ kernel_index];
					gs_temp *=
							intermediate_variables[(update_order + 2) % order][index_2
									+ kernel_index];
					gs += gs_temp;
				}

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_dimen * order_3;
					type_of_data p_a_gs =
							parameter_a[(update_order + 3) % order][index_3
									+ lane_id] * gs;

					int temp = core;
					while (temp != 1) {
						temp /= 2;
						p_a_gs += __shfl_down_sync(mask, p_a_gs, temp);
					}

					p_a_gs = __shfl_sync(mask, p_a_gs,
							(local_id % local) * core);

					p_a_gs -= value_train_device[update_order][order_index_3];

					parameter_a[(update_order + 3) % order][index_3 + lane_id] -=
							learn_rate_a
									* (p_a_gs * gs
											+ lambda_a
													* parameter_a[(update_order
															+ 3) % order][index_3
															+ lane_id]);
				}
			}
		}
	}
}

__global__ void Update_Parameter_A_SGD_Order_5(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device, type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		const type_of_data learn_rate_a, const type_of_data lambda_a,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					type_of_data gs = 0.0;

					for (int kernel_index = 0; kernel_index < core_kernel;
							kernel_index++) {
						type_of_data gs_temp = parameter_b[(update_order + 4)
								% order][kernel_index * core_dimen + lane_id];
						gs_temp *= intermediate_variables[update_order][index_0
								+ kernel_index];
						gs_temp *= intermediate_variables[(update_order + 1)
								% order][index_1 + kernel_index];
						gs_temp *= intermediate_variables[(update_order + 2)
								% order][index_2 + kernel_index];
						gs_temp *= intermediate_variables[(update_order + 3)
								% order][index_3 + kernel_index];
						gs += gs_temp;
					}

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_dimen * order_4;

						type_of_data p_a_gs = parameter_a[(update_order + 4)
								% order][index_4 + lane_id] * gs;

						int temp = core;
						while (temp != 1) {
							temp /= 2;
							p_a_gs += __shfl_down_sync(mask, p_a_gs, temp);
						}

						p_a_gs = __shfl_sync(mask, p_a_gs,
								(local_id % local) * core);

						p_a_gs -=
								value_train_device[update_order][order_index_4];

						parameter_a[(update_order + 4) % order][index_4
								+ lane_id] -= learn_rate_a
								* (p_a_gs * gs
										+ lambda_a
												* parameter_a[(update_order + 4)
														% order][index_4
														+ lane_id]);
					}
				}
			}
		}
	}
}

__global__ void Update_Parameter_A_SGD_Order_6(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device, type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		const type_of_data learn_rate_a, const type_of_data lambda_a,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int start_4 =
								ptr_train_device[update_order][5][order_index_4];
						int end_4 =
								ptr_train_device[update_order][5][order_index_4
										+ 1];
						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_dimen * order_4;

						type_of_data gs = 0.0;

						for (int kernel_index = 0; kernel_index < core_kernel;
								kernel_index++) {

							type_of_data gs_temp =
									parameter_b[(update_order + 5) % order][kernel_index
											* core_dimen + lane_id];
							gs_temp *=
									intermediate_variables[update_order][index_0
											+ kernel_index];
							gs_temp *= intermediate_variables[(update_order + 1)
									% order][index_1 + kernel_index];
							gs_temp *= intermediate_variables[(update_order + 2)
									% order][index_2 + kernel_index];
							gs_temp *= intermediate_variables[(update_order + 3)
									% order][index_3 + kernel_index];
							gs_temp *= intermediate_variables[(update_order + 4)
									% order][index_4 + kernel_index];
							gs += gs_temp;
						}

						for (int order_index_5 = start_4; order_index_5 < end_4;
								order_index_5++) {

							int order_5 =
									idx_train_device[update_order][5][order_index_5];
							int index_5 = core_dimen * order_5;

							type_of_data p_a_gs = parameter_a[(update_order + 5)
									% order][index_5 + lane_id] * gs;

							int temp = core;
							while (temp != 1) {
								temp /= 2;
								p_a_gs += __shfl_down_sync(mask, p_a_gs, temp);
							}

							p_a_gs = __shfl_sync(mask, p_a_gs,
									(local_id % local) * core);

							p_a_gs -=
									value_train_device[update_order][order_index_5];

							parameter_a[(update_order + 5) % order][index_5
									+ lane_id] -=
									learn_rate_a
											* (p_a_gs * gs
													+ lambda_a
															* parameter_a[(update_order
																	+ 5) % order][index_5
																	+ lane_id]);
						}
					}
				}
			}
		}
	}
}

__global__ void Update_Parameter_A_SGD_Order_7(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device, type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		const type_of_data learn_rate_a, const type_of_data lambda_a,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int start_4 =
								ptr_train_device[update_order][5][order_index_4];
						int end_4 =
								ptr_train_device[update_order][5][order_index_4
										+ 1];
						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_kernel * order_4;

						for (int order_index_5 = start_4; order_index_5 < end_4;
								order_index_5++) {

							int start_5 =
									ptr_train_device[update_order][6][order_index_5];
							int end_5 =
									ptr_train_device[update_order][6][order_index_5
											+ 1];
							int order_5 =
									idx_train_device[update_order][5][order_index_5];
							int index_5 = core_kernel * order_5;

							type_of_data gs = 0.0;

							for (int kernel_index = 0;
									kernel_index < core_kernel;
									kernel_index++) {

								type_of_data gs_temp = parameter_b[(update_order
										+ 6) % order][kernel_index * core_dimen
										+ lane_id];
								gs_temp *=
										intermediate_variables[update_order][index_0
												+ kernel_index];
								gs_temp *= intermediate_variables[(update_order
										+ 1) % order][index_1 + kernel_index];
								gs_temp *= intermediate_variables[(update_order
										+ 2) % order][index_2 + kernel_index];
								gs_temp *= intermediate_variables[(update_order
										+ 3) % order][index_3 + kernel_index];
								gs_temp *= intermediate_variables[(update_order
										+ 4) % order][index_4 + kernel_index];
								gs_temp *= intermediate_variables[(update_order
										+ 5) % order][index_5 + kernel_index];
								gs += gs_temp;
							}

							for (int order_index_6 = start_5;
									order_index_6 < end_5; order_index_6++) {

								int order_6 =
										idx_train_device[update_order][6][order_index_6];
								int index_6 = core_dimen * order_6;
								type_of_data p_a_gs = parameter_a[(update_order
										+ 6) % order][index_6 + lane_id] * gs;

								int temp = core;
								while (temp != 1) {
									temp /= 2;
									p_a_gs += __shfl_down_sync(mask, p_a_gs,
											temp);
								}

								p_a_gs = __shfl_sync(mask, p_a_gs,
										(local_id % local) * core);

								p_a_gs -=
										value_train_device[update_order][order_index_6];

								parameter_a[(update_order + 6) % order][index_6
										+ lane_id] -=
										learn_rate_a
												* (p_a_gs * gs
														+ lambda_a
																* parameter_a[(update_order
																		+ 6)
																		% order][index_6
																		+ lane_id]);
							}
						}
					}
				}
			}
		}
	}
}

__global__ void Update_Parameter_A_SGD_Order_8(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device, type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		const type_of_data learn_rate_a, const type_of_data lambda_a,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int start_4 =
								ptr_train_device[update_order][5][order_index_4];
						int end_4 =
								ptr_train_device[update_order][5][order_index_4
										+ 1];
						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_kernel * order_4;

						for (int order_index_5 = start_4; order_index_5 < end_4;
								order_index_5++) {

							int start_5 =
									ptr_train_device[update_order][6][order_index_5];
							int end_5 =
									ptr_train_device[update_order][6][order_index_5
											+ 1];
							int order_5 =
									idx_train_device[update_order][5][order_index_5];
							int index_5 = core_kernel * order_5;

							for (int order_index_6 = start_5;
									order_index_6 < end_5; order_index_6++) {

								int start_6 =
										ptr_train_device[update_order][7][order_index_6];
								int end_6 =
										ptr_train_device[update_order][7][order_index_6
												+ 1];
								int order_6 =
										idx_train_device[update_order][6][order_index_6];
								int index_6 = core_kernel * order_6;

								type_of_data gs = 0.0;

								for (int kernel_index = 0;
										kernel_index < core_kernel;
										kernel_index++) {

									type_of_data gs_temp =
											parameter_b[(update_order + 7)
													% order][kernel_index
													* core_dimen + lane_id];
									gs_temp *=
											intermediate_variables[update_order][index_0
													+ kernel_index];
									gs_temp *=
											intermediate_variables[(update_order
													+ 1) % order][index_1
													+ kernel_index];
									gs_temp *=
											intermediate_variables[(update_order
													+ 2) % order][index_2
													+ kernel_index];
									gs_temp *=
											intermediate_variables[(update_order
													+ 3) % order][index_3
													+ kernel_index];
									gs_temp *=
											intermediate_variables[(update_order
													+ 4) % order][index_4
													+ kernel_index];
									gs_temp *=
											intermediate_variables[(update_order
													+ 5) % order][index_5
													+ kernel_index];
									gs_temp *=
											intermediate_variables[(update_order
													+ 6) % order][index_6
													+ kernel_index];
									gs += gs_temp;
								}

								for (int order_index_7 = start_6;
										order_index_7 < end_6;
										order_index_7++) {

									int order_7 =
											idx_train_device[update_order][7][order_index_7];
									int index_7 = core_dimen * order_7;

									type_of_data p_a_gs =
											parameter_a[(update_order + 7)
													% order][index_7 + lane_id]
													* gs;

									int temp = core;
									while (temp != 1) {
										temp /= 2;
										p_a_gs += __shfl_down_sync(mask, p_a_gs,
												temp);
									}

									p_a_gs = __shfl_sync(mask, p_a_gs,
											(local_id % local) * core);

									p_a_gs -=
											value_train_device[update_order][order_index_7];

									parameter_a[(update_order + 7) % order][index_7
											+ lane_id] -=
											learn_rate_a
													* (p_a_gs * gs
															+ lambda_a
																	* parameter_a[(update_order
																			+ 7)
																			% order][index_7
																			+ lane_id]);
								}
							}
						}
					}
				}
			}
		}
	}
}

__global__ void Update_Parameter_A_SGD_Order_9(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device, type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		const type_of_data learn_rate_a, const type_of_data lambda_a,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int start_4 =
								ptr_train_device[update_order][5][order_index_4];
						int end_4 =
								ptr_train_device[update_order][5][order_index_4
										+ 1];
						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_kernel * order_4;

						for (int order_index_5 = start_4; order_index_5 < end_4;
								order_index_5++) {

							int start_5 =
									ptr_train_device[update_order][6][order_index_5];
							int end_5 =
									ptr_train_device[update_order][6][order_index_5
											+ 1];
							int order_5 =
									idx_train_device[update_order][5][order_index_5];
							int index_5 = core_kernel * order_5;

							for (int order_index_6 = start_5;
									order_index_6 < end_5; order_index_6++) {

								int start_6 =
										ptr_train_device[update_order][7][order_index_6];
								int end_6 =
										ptr_train_device[update_order][7][order_index_6
												+ 1];
								int order_6 =
										idx_train_device[update_order][6][order_index_6];
								int index_6 = core_kernel * order_6;

								for (int order_index_7 = start_6;
										order_index_7 < end_6;
										order_index_7++) {

									int start_7 =
											ptr_train_device[update_order][8][order_index_7];
									int end_7 =
											ptr_train_device[update_order][8][order_index_7
													+ 1];
									int order_7 =
											idx_train_device[update_order][7][order_index_7];
									int index_7 = core_kernel * order_7;

									type_of_data gs = 0.0;

									for (int kernel_index = 0;
											kernel_index < core_kernel;
											kernel_index++) {

										type_of_data gs_temp =
												parameter_b[(update_order + 8)
														% order][kernel_index
														* core_dimen + lane_id];
										gs_temp *=
												intermediate_variables[update_order][index_0
														+ kernel_index];
										gs_temp *=
												intermediate_variables[(update_order
														+ 1) % order][index_1
														+ kernel_index];
										gs_temp *=
												intermediate_variables[(update_order
														+ 2) % order][index_2
														+ kernel_index];
										gs_temp *=
												intermediate_variables[(update_order
														+ 3) % order][index_3
														+ kernel_index];
										gs_temp *=
												intermediate_variables[(update_order
														+ 4) % order][index_4
														+ kernel_index];
										gs_temp *=
												intermediate_variables[(update_order
														+ 5) % order][index_5
														+ kernel_index];
										gs_temp *=
												intermediate_variables[(update_order
														+ 6) % order][index_6
														+ kernel_index];
										gs_temp *=
												intermediate_variables[(update_order
														+ 7) % order][index_7
														+ kernel_index];
										gs += gs_temp;
									}

									for (int order_index_8 = start_7;
											order_index_8 < end_7;
											order_index_8++) {

										int order_8 =
												idx_train_device[update_order][8][order_index_8];
										int index_8 = core_dimen * order_8;
										type_of_data p_a_gs =
												parameter_a[(update_order + 8)
														% order][index_8
														+ lane_id] * gs;

										int temp = core;
										while (temp != 1) {
											temp /= 2;
											p_a_gs += __shfl_down_sync(
											mask, p_a_gs, temp);
										}

										p_a_gs = __shfl_sync(mask, p_a_gs,
												(local_id % local) * core);

										p_a_gs -=
												value_train_device[update_order][order_index_8];

										parameter_a[(update_order + 8) % order][index_8
												+ lane_id] -=
												learn_rate_a
														* (p_a_gs * gs
																+ lambda_a
																		* parameter_a[(update_order
																				+ 8)
																				% order][index_8
																				+ lane_id]);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

__global__ void Update_Parameter_A_SGD_Order_10(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device, type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		const type_of_data learn_rate_a, const type_of_data lambda_a,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int start_4 =
								ptr_train_device[update_order][5][order_index_4];
						int end_4 =
								ptr_train_device[update_order][5][order_index_4
										+ 1];
						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_kernel * order_4;

						for (int order_index_5 = start_4; order_index_5 < end_4;
								order_index_5++) {

							int start_5 =
									ptr_train_device[update_order][6][order_index_5];
							int end_5 =
									ptr_train_device[update_order][6][order_index_5
											+ 1];
							int order_5 =
									idx_train_device[update_order][5][order_index_5];
							int index_5 = core_kernel * order_5;

							for (int order_index_6 = start_5;
									order_index_6 < end_5; order_index_6++) {

								int start_6 =
										ptr_train_device[update_order][7][order_index_6];
								int end_6 =
										ptr_train_device[update_order][7][order_index_6
												+ 1];
								int order_6 =
										idx_train_device[update_order][6][order_index_6];
								int index_6 = core_kernel * order_6;

								for (int order_index_7 = start_6;
										order_index_7 < end_6;
										order_index_7++) {

									int start_7 =
											ptr_train_device[update_order][8][order_index_7];
									int end_7 =
											ptr_train_device[update_order][8][order_index_7
													+ 1];
									int order_7 =
											idx_train_device[update_order][7][order_index_7];
									int index_7 = core_kernel * order_7;

									for (int order_index_8 = start_7;
											order_index_8 < end_7;
											order_index_8++) {

										int start_8 =
												ptr_train_device[update_order][9][order_index_8];
										int end_8 =
												ptr_train_device[update_order][9][order_index_8
														+ 1];
										int order_8 =
												idx_train_device[update_order][8][order_index_8];
										int index_8 = core_kernel * order_8;

										type_of_data gs = 0.0;

										for (int kernel_index = 0;
												kernel_index < core_kernel;
												kernel_index++) {

											type_of_data gs_temp =
													parameter_b[(update_order
															+ 8) % order][kernel_index
															* core_dimen
															+ lane_id];
											gs_temp *=
													intermediate_variables[update_order][index_0
															+ kernel_index];
											gs_temp *=
													intermediate_variables[(update_order
															+ 1) % order][index_1
															+ kernel_index];
											gs_temp *=
													intermediate_variables[(update_order
															+ 2) % order][index_2
															+ kernel_index];
											gs_temp *=
													intermediate_variables[(update_order
															+ 3) % order][index_3
															+ kernel_index];
											gs_temp *=
													intermediate_variables[(update_order
															+ 4) % order][index_4
															+ kernel_index];
											gs_temp *=
													intermediate_variables[(update_order
															+ 5) % order][index_5
															+ kernel_index];
											gs_temp *=
													intermediate_variables[(update_order
															+ 6) % order][index_6
															+ kernel_index];
											gs_temp *=
													intermediate_variables[(update_order
															+ 7) % order][index_7
															+ kernel_index];
											gs_temp *=
													intermediate_variables[(update_order
															+ 8) % order][index_8
															+ kernel_index];
											gs += gs_temp;
										}

										for (int order_index_9 = start_8;
												order_index_9 < end_8;
												order_index_9++) {

											int order_9 =
													idx_train_device[update_order][9][order_index_9];
											int index_9 = core_dimen * order_9;

											type_of_data p_a_gs =
													parameter_a[(update_order
															+ 9) % order][index_9
															+ lane_id] * gs;

											int temp = core;
											while (temp != 1) {
												temp /= 2;
												p_a_gs += __shfl_down_sync(
												mask, p_a_gs, temp);
											}

											p_a_gs = __shfl_sync(mask, p_a_gs,
													(local_id % local) * core);

											p_a_gs -=
													value_train_device[update_order][order_index_9];

											parameter_a[(update_order + 9)
													% order][index_9 + lane_id] -=
													learn_rate_a
															* (p_a_gs * gs
																	+ lambda_a
																			* parameter_a[(update_order
																					+ 9)
																					% order][index_9
																					+ lane_id]);
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

void Update_Parameter_A(const int order, int *dimen, const int core_kernel,
		const int core_dimen, int nnz_train, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a_device,
		type_of_data **parameter_b_device, const type_of_data learn_rate_a,
		const type_of_data lambda_a, type_of_data **intermediate_variables) {

	for (int update_order = 0; update_order < order; update_order++) {

		if (order == 3) {
			Update_Parameter_A_SGD_Order_3 <<<grid_size, block_size
			>>>(order, update_order, core_kernel, core_dimen, nnz_train,
					idx_train_len_device, ptr_train_device, idx_train_device,
					value_train_device, parameter_a_device, parameter_b_device,
					learn_rate_a, lambda_a, intermediate_variables);
			cudaDeviceSynchronize();
		} else if (order == 4) {
			Update_Parameter_A_SGD_Order_4 <<<grid_size, block_size
			>>>(order, update_order, core_kernel, core_dimen, nnz_train,
					idx_train_len_device, ptr_train_device, idx_train_device,
					value_train_device, parameter_a_device, parameter_b_device,
					learn_rate_a, lambda_a, intermediate_variables);
			cudaDeviceSynchronize();

		} else if (order == 5) {
			Update_Parameter_A_SGD_Order_5 <<<grid_size, block_size
			>>>(order, update_order, core_kernel, core_dimen, nnz_train,
					idx_train_len_device, ptr_train_device, idx_train_device,
					value_train_device, parameter_a_device, parameter_b_device,
					learn_rate_a, lambda_a, intermediate_variables);
			cudaDeviceSynchronize();

		} else if (order == 6) {
			Update_Parameter_A_SGD_Order_6 <<<grid_size, block_size
			>>>(order, update_order, core_kernel, core_dimen, nnz_train,
					idx_train_len_device, ptr_train_device, idx_train_device,
					value_train_device, parameter_a_device, parameter_b_device,
					learn_rate_a, lambda_a, intermediate_variables);
			cudaDeviceSynchronize();

		} else if (order == 7) {
			Update_Parameter_A_SGD_Order_7 <<<grid_size, block_size
			>>>(order, update_order, core_kernel, core_dimen, nnz_train,
					idx_train_len_device, ptr_train_device, idx_train_device,
					value_train_device, parameter_a_device, parameter_b_device,
					learn_rate_a, lambda_a, intermediate_variables);
			cudaDeviceSynchronize();

		} else if (order == 8) {
			Update_Parameter_A_SGD_Order_8 <<<grid_size, block_size
			>>>(order, update_order, core_kernel, core_dimen, nnz_train,
					idx_train_len_device, ptr_train_device, idx_train_device,
					value_train_device, parameter_a_device, parameter_b_device,
					learn_rate_a, lambda_a, intermediate_variables);
			cudaDeviceSynchronize();

		} else if (order == 9) {
			Update_Parameter_A_SGD_Order_9 <<<grid_size, block_size
			>>>(order, update_order, core_kernel, core_dimen, nnz_train,
					idx_train_len_device, ptr_train_device, idx_train_device,
					value_train_device, parameter_a_device, parameter_b_device,
					learn_rate_a, lambda_a, intermediate_variables);
			cudaDeviceSynchronize();

		} else if (order == 10) {
			Update_Parameter_A_SGD_Order_10 <<<grid_size, block_size
			>>>(order, update_order, core_kernel, core_dimen, nnz_train,
					idx_train_len_device, ptr_train_device, idx_train_device,
					value_train_device, parameter_a_device, parameter_b_device,
					learn_rate_a, lambda_a, intermediate_variables);
			cudaDeviceSynchronize();
		}

		int fact_order = (update_order + order - 1) % order;

		Calculate_Intermediate_Variables
				<<<dimen[fact_order] / block_size * core_dimen, block_size>>>(order,
				core_kernel, core_dimen, parameter_a_device, parameter_b_device,
				fact_order, dimen[fact_order], intermediate_variables);
		cudaDeviceSynchronize();

	}

}

__global__ void Update_Parameter_B_CSF_Gradient_Order_3(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device, type_of_data **parameter_a,
		type_of_data **parameter_b, type_of_data *b_sum,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local = warp_size / core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data shared[];
	type_of_data *ho_shared = shared;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			type_of_data gs = 0.0;

			for (int kernel_index = 0; kernel_index < core_kernel;
					kernel_index++) {

				ho_shared[kernel_index * block_size + threadIdx.x] = 1.0f;
				ho_shared[kernel_index * block_size + threadIdx.x] *=
						intermediate_variables[update_order][index_0
								+ kernel_index];
				ho_shared[kernel_index * block_size + threadIdx.x] *=
						intermediate_variables[(update_order + 1) % order][index_1
								+ kernel_index];
				type_of_data gs_temp =
						parameter_b[(update_order + 2) % order][kernel_index
								* core_dimen + lane_id];
				gs_temp *= ho_shared[kernel_index * block_size + threadIdx.x];
				gs += gs_temp;
			}

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_dimen * order_2;

				type_of_data parameter_a_temp = parameter_a[(update_order + 2)
						% order][index_2 + lane_id];

				type_of_data p_a_gs = parameter_a_temp * gs;

				int temp = core;
				while (temp != 1) {
					temp /= 2;
					p_a_gs += __shfl_down_sync(mask, p_a_gs, temp);
				}

				p_a_gs = __shfl_sync(mask, p_a_gs, (local_id % local) * core);

				p_a_gs -= value_train_device[update_order][order_index_2];

				for (int core_kernel_index = 0; core_kernel_index < core_kernel;
						core_kernel_index++) {
					atomicAdd(
							&b_sum[(order_index_2 % sum_size) * core_kernel
									* core_dimen
									+ core_kernel_index * core_dimen + lane_id],
							p_a_gs

							* parameter_a_temp
									* ho_shared[core_kernel_index * block_size
											+ threadIdx.x]);
				}

			}
		}
	}
}

__global__ void Update_Parameter_B_CSF_Gradient_Order_4(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device, type_of_data **parameter_a,
		type_of_data **parameter_b, type_of_data *b_sum,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local = warp_size / core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data shared[];
	type_of_data *ho_shared = shared;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				type_of_data gs = 0.0;

				for (int kernel_index = 0; kernel_index < core_kernel;
						kernel_index++) {

					ho_shared[kernel_index * block_size + threadIdx.x] = 1.0f;
					ho_shared[kernel_index * block_size + threadIdx.x] *=
							intermediate_variables[update_order][index_0
									+ kernel_index];
					ho_shared[kernel_index * block_size + threadIdx.x] *=
							intermediate_variables[(update_order + 1) % order][index_1
									+ kernel_index];
					ho_shared[kernel_index * block_size + threadIdx.x] *=
							intermediate_variables[(update_order + 2) % order][index_2
									+ kernel_index];
					type_of_data gs_temp = parameter_b[(update_order + 3)
							% order][kernel_index * core_dimen + lane_id];
					gs_temp *=
							ho_shared[kernel_index * block_size + threadIdx.x];
					gs += gs_temp;

				}

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_dimen * order_3;

					type_of_data parameter_a_temp = parameter_a[(update_order
							+ 3) % order][index_3 + lane_id];
					type_of_data p_a_gs = parameter_a_temp * gs;

					int temp = core;
					while (temp != 1) {
						temp /= 2;
						p_a_gs += __shfl_down_sync(mask, p_a_gs, temp);
					}

					p_a_gs = __shfl_sync(mask, p_a_gs,
							(local_id % local) * core);

					p_a_gs -= value_train_device[update_order][order_index_3];

					for (int core_kernel_index = 0;
							core_kernel_index < core_kernel;
							core_kernel_index++) {
						atomicAdd(
								&b_sum[(order_index_3 % sum_size) * core_kernel
										* core_dimen
										+ core_kernel_index * core_dimen
										+ lane_id],
								p_a_gs * parameter_a_temp
										* ho_shared[core_kernel_index
												* block_size + threadIdx.x]);
					}

				}
			}
		}
	}
}

__global__ void Update_Parameter_B_CSF_Gradient_Order_5(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device, type_of_data **parameter_a,
		type_of_data **parameter_b, type_of_data *b_sum,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local = warp_size / core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data shared[];
	type_of_data *ho_shared = shared;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					type_of_data gs = 0.0;

					for (int kernel_index = 0; kernel_index < core_kernel;
							kernel_index++) {

						ho_shared[kernel_index * block_size + threadIdx.x] =
								1.0f;
						ho_shared[kernel_index * block_size + threadIdx.x] *=
								intermediate_variables[update_order][index_0
										+ kernel_index];
						ho_shared[kernel_index * block_size + threadIdx.x] *=
								intermediate_variables[(update_order + 1)
										% order][index_1 + kernel_index];
						ho_shared[kernel_index * block_size + threadIdx.x] *=
								intermediate_variables[(update_order + 2)
										% order][index_2 + kernel_index];
						ho_shared[kernel_index * block_size + threadIdx.x] *=
								intermediate_variables[(update_order + 3)
										% order][index_3 + kernel_index];

						type_of_data gs_temp = parameter_b[(update_order + 4)
								% order][kernel_index * core_dimen + lane_id];
						gs_temp *= ho_shared[kernel_index * block_size
								+ threadIdx.x];
						gs += gs_temp;

					}

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_dimen * order_4;

						type_of_data parameter_a_temp =
								parameter_a[(update_order + 4) % order][index_4
										+ lane_id];

						type_of_data p_a_gs = parameter_a_temp * gs;

						int temp = core;
						while (temp != 1) {
							temp /= 2;
							p_a_gs += __shfl_down_sync(mask, p_a_gs, temp);
						}

						p_a_gs = __shfl_sync(mask, p_a_gs,
								(local_id % local) * core);

						p_a_gs -=
								value_train_device[update_order][order_index_4];

						for (int core_kernel_index = 0;
								core_kernel_index < core_kernel;
								core_kernel_index++) {
							atomicAdd(
									&b_sum[(order_index_4 % sum_size)
											* core_kernel * core_dimen
											+ core_kernel_index * core_dimen
											+ lane_id],
									p_a_gs * parameter_a_temp
											* ho_shared[core_kernel_index
													* block_size + threadIdx.x]);
						}
					}
				}
			}
		}
	}
}

__global__ void Update_Parameter_B_CSF_Gradient_Order_6(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device, type_of_data **parameter_a,
		type_of_data **parameter_b, type_of_data *b_sum,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local = warp_size / core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data shared[];
	type_of_data *ho_shared = shared;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int start_4 =
								ptr_train_device[update_order][5][order_index_4];
						int end_4 =
								ptr_train_device[update_order][5][order_index_4
										+ 1];
						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_kernel * order_4;

						type_of_data gs = 0.0;

						for (int kernel_index = 0; kernel_index < core_kernel;
								kernel_index++) {

							ho_shared[kernel_index * block_size + threadIdx.x] =
									1.0f;
							ho_shared[kernel_index * block_size + threadIdx.x] *=
									intermediate_variables[update_order][index_0
											+ kernel_index];
							ho_shared[kernel_index * block_size + threadIdx.x] *=
									intermediate_variables[(update_order + 1)
											% order][index_1 + kernel_index];
							ho_shared[kernel_index * block_size + threadIdx.x] *=
									intermediate_variables[(update_order + 2)
											% order][index_2 + kernel_index];
							ho_shared[kernel_index * block_size + threadIdx.x] *=
									intermediate_variables[(update_order + 3)
											% order][index_3 + kernel_index];
							ho_shared[kernel_index * block_size + threadIdx.x] *=
									intermediate_variables[(update_order + 4)
											% order][index_4 + kernel_index];
							type_of_data gs_temp =
									parameter_b[(update_order + 5) % order][kernel_index
											* core_dimen + lane_id];
							gs_temp *= ho_shared[kernel_index * block_size
									+ threadIdx.x];
							gs += gs_temp;
						}

						for (int order_index_5 = start_4; order_index_5 < end_4;
								order_index_5++) {

							int order_5 =
									idx_train_device[update_order][5][order_index_5];
							int index_5 = core_dimen * order_5;

							type_of_data parameter_a_temp =
									parameter_a[(update_order + 5) % order][index_5
											+ lane_id];

							type_of_data p_a_gs = parameter_a_temp * gs;

							int temp = core;
							while (temp != 1) {
								temp /= 2;
								p_a_gs += __shfl_down_sync(mask, p_a_gs, temp);
							}

							p_a_gs = __shfl_sync(mask, p_a_gs,
									(local_id % local) * core);
							p_a_gs -=
									value_train_device[update_order][order_index_5];

							for (int core_kernel_index = 0;
									core_kernel_index < core_kernel;
									core_kernel_index++) {
								atomicAdd(
										&b_sum[(order_index_5 % sum_size)
												* core_kernel * core_dimen
												+ core_kernel_index * core_dimen
												+ lane_id],
										p_a_gs * parameter_a_temp
												* ho_shared[core_kernel_index
														* block_size
														+ threadIdx.x]);
							}
						}
					}
				}
			}
		}
	}
}

__global__ void Update_Parameter_B_CSF_Gradient_Order_7(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device, type_of_data **parameter_a,
		type_of_data **parameter_b, type_of_data *b_sum,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local = warp_size / core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data shared[];
	type_of_data *ho_shared = shared;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int start_4 =
								ptr_train_device[update_order][5][order_index_4];
						int end_4 =
								ptr_train_device[update_order][5][order_index_4
										+ 1];
						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_kernel * order_4;

						for (int order_index_5 = start_4; order_index_5 < end_4;
								order_index_5++) {

							int start_5 =
									ptr_train_device[update_order][6][order_index_5];
							int end_5 =
									ptr_train_device[update_order][6][order_index_5
											+ 1];
							int order_5 =
									idx_train_device[update_order][5][order_index_5];
							int index_5 = core_kernel * order_5;

							type_of_data gs = 0.0;

							for (int kernel_index = 0;
									kernel_index < core_kernel;
									kernel_index++) {

								ho_shared[kernel_index * block_size
										+ threadIdx.x] = 1.0f;
								ho_shared[kernel_index * block_size
										+ threadIdx.x] *=
										intermediate_variables[update_order][index_0
												+ kernel_index];
								ho_shared[kernel_index * block_size
										+ threadIdx.x] *=
										intermediate_variables[(update_order + 1)
												% order][index_1 + kernel_index];
								ho_shared[kernel_index * block_size
										+ threadIdx.x] *=
										intermediate_variables[(update_order + 2)
												% order][index_2 + kernel_index];
								ho_shared[kernel_index * block_size
										+ threadIdx.x] *=
										intermediate_variables[(update_order + 3)
												% order][index_3 + kernel_index];
								ho_shared[kernel_index * block_size
										+ threadIdx.x] *=
										intermediate_variables[(update_order + 4)
												% order][index_4 + kernel_index];
								ho_shared[kernel_index * block_size
										+ threadIdx.x] *=
										intermediate_variables[(update_order + 5)
												% order][index_5 + kernel_index];
								type_of_data gs_temp = parameter_b[(update_order
										+ 6) % order][kernel_index * core_dimen
										+ lane_id];
								gs_temp *= ho_shared[kernel_index * block_size
										+ threadIdx.x];
								gs += gs_temp;
							}
							for (int order_index_6 = start_5;
									order_index_6 < end_5; order_index_6++) {

								int order_6 =
										idx_train_device[update_order][6][order_index_6];
								int index_6 = core_dimen * order_6;
								type_of_data parameter_a_temp =
										parameter_a[(update_order + 6) % order][index_6
												+ lane_id];

								type_of_data p_a_gs = parameter_a_temp * gs;

								int temp = core;
								while (temp != 1) {
									temp /= 2;
									p_a_gs += __shfl_down_sync(mask, p_a_gs,
											temp);
								}

								p_a_gs = __shfl_sync(mask, p_a_gs,
										(local_id % local) * core);
								p_a_gs -=
										value_train_device[update_order][order_index_6];

								for (int core_kernel_index = 0;
										core_kernel_index < core_kernel;
										core_kernel_index++) {
									atomicAdd(
											&b_sum[(order_index_6 % sum_size)
													* core_kernel * core_dimen
													+ core_kernel_index
															* core_dimen
													+ lane_id],
											p_a_gs

											* parameter_a_temp
													* ho_shared[core_kernel_index
															* block_size
															+ threadIdx.x]);
								}
							}
						}
					}
				}
			}
		}
	}
}

__global__ void Update_Parameter_B_CSF_Gradient_Order_8(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device, type_of_data **parameter_a,
		type_of_data **parameter_b, type_of_data *b_sum,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local = warp_size / core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data shared[];
	type_of_data *ho_shared = shared;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int start_4 =
								ptr_train_device[update_order][5][order_index_4];
						int end_4 =
								ptr_train_device[update_order][5][order_index_4
										+ 1];
						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_kernel * order_4;

						for (int order_index_5 = start_4; order_index_5 < end_4;
								order_index_5++) {

							int start_5 =
									ptr_train_device[update_order][6][order_index_5];
							int end_5 =
									ptr_train_device[update_order][6][order_index_5
											+ 1];
							int order_5 =
									idx_train_device[update_order][5][order_index_5];
							int index_5 = core_kernel * order_5;

							for (int order_index_6 = start_5;
									order_index_6 < end_5; order_index_6++) {

								int start_6 =
										ptr_train_device[update_order][7][order_index_6];
								int end_6 =
										ptr_train_device[update_order][7][order_index_6
												+ 1];
								int order_6 =
										idx_train_device[update_order][6][order_index_6];
								int index_6 = core_kernel * order_6;

								type_of_data gs = 0.0;

								for (int kernel_index = 0;
										kernel_index < core_kernel;
										kernel_index++) {

									ho_shared[kernel_index * block_size
											+ threadIdx.x] = 1.0f;
									ho_shared[kernel_index * block_size
											+ threadIdx.x] *=
											intermediate_variables[update_order][index_0
													+ kernel_index];
									ho_shared[kernel_index * block_size
											+ threadIdx.x] *=
											intermediate_variables[(update_order
													+ 1) % order][index_1
													+ kernel_index];
									ho_shared[kernel_index * block_size
											+ threadIdx.x] *=
											intermediate_variables[(update_order
													+ 2) % order][index_2
													+ kernel_index];
									ho_shared[kernel_index * block_size
											+ threadIdx.x] *=
											intermediate_variables[(update_order
													+ 3) % order][index_3
													+ kernel_index];
									ho_shared[kernel_index * block_size
											+ threadIdx.x] *=
											intermediate_variables[(update_order
													+ 4) % order][index_4
													+ kernel_index];
									ho_shared[kernel_index * block_size
											+ threadIdx.x] *=
											intermediate_variables[(update_order
													+ 5) % order][index_5
													+ kernel_index];
									ho_shared[kernel_index * block_size
											+ threadIdx.x] *=
											intermediate_variables[(update_order
													+ 6) % order][index_6
													+ kernel_index];
									type_of_data gs_temp =
											parameter_b[(update_order + 7)
													% order][kernel_index
													* core_dimen + lane_id];
									gs_temp *= ho_shared[kernel_index
											* block_size + threadIdx.x];
									gs += gs_temp;

								}

								for (int order_index_7 = start_6;
										order_index_7 < end_6;
										order_index_7++) {

									int order_7 =
											idx_train_device[update_order][7][order_index_7];
									int index_7 = core_dimen * order_7;

									type_of_data parameter_a_temp =
											parameter_a[(update_order + 7)
													% order][index_7 + lane_id];

									type_of_data p_a_gs = parameter_a_temp * gs;

									int temp = core;
									while (temp != 1) {
										temp /= 2;
										p_a_gs += __shfl_down_sync(mask, p_a_gs,
												temp);
									}

									p_a_gs = __shfl_sync(mask, p_a_gs,
											(local_id % local) * core);
									p_a_gs -=
											value_train_device[update_order][order_index_7];

									for (int core_kernel_index = 0;
											core_kernel_index < core_kernel;
											core_kernel_index++) {
										atomicAdd(
												&b_sum[(order_index_7 % sum_size)
														* core_kernel
														* core_dimen
														+ core_kernel_index
																* core_dimen
														+ lane_id],
												p_a_gs * parameter_a_temp
														* ho_shared[core_kernel_index
																* block_size
																+ threadIdx.x]);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

__global__ void Update_Parameter_B_CSF_Gradient_Order_9(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device, type_of_data **parameter_a,
		type_of_data **parameter_b, type_of_data *b_sum,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local = warp_size / core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data shared[];
	type_of_data *ho_shared = shared;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int start_4 =
								ptr_train_device[update_order][5][order_index_4];
						int end_4 =
								ptr_train_device[update_order][5][order_index_4
										+ 1];
						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_kernel * order_4;

						for (int order_index_5 = start_4; order_index_5 < end_4;
								order_index_5++) {

							int start_5 =
									ptr_train_device[update_order][6][order_index_5];
							int end_5 =
									ptr_train_device[update_order][6][order_index_5
											+ 1];
							int order_5 =
									idx_train_device[update_order][5][order_index_5];
							int index_5 = core_kernel * order_5;

							for (int order_index_6 = start_5;
									order_index_6 < end_5; order_index_6++) {

								int start_6 =
										ptr_train_device[update_order][7][order_index_6];
								int end_6 =
										ptr_train_device[update_order][7][order_index_6
												+ 1];
								int order_6 =
										idx_train_device[update_order][6][order_index_6];
								int index_6 = core_kernel * order_6;

								for (int order_index_7 = start_6;
										order_index_7 < end_6;
										order_index_7++) {

									int start_7 =
											ptr_train_device[update_order][8][order_index_7];
									int end_7 =
											ptr_train_device[update_order][8][order_index_7
													+ 1];
									int order_7 =
											idx_train_device[update_order][7][order_index_7];
									int index_7 = core_kernel * order_7;

									type_of_data gs = 0.0;

									for (int kernel_index = 0;
											kernel_index < core_kernel;
											kernel_index++) {

										ho_shared[kernel_index * block_size
												+ threadIdx.x] = 1.0f;
										ho_shared[kernel_index * block_size
												+ threadIdx.x] *=
												intermediate_variables[update_order][index_0
														+ kernel_index];
										ho_shared[kernel_index * block_size
												+ threadIdx.x] *=
												intermediate_variables[(update_order
														+ 1) % order][index_1
														+ kernel_index];
										ho_shared[kernel_index * block_size
												+ threadIdx.x] *=
												intermediate_variables[(update_order
														+ 2) % order][index_2
														+ kernel_index];
										ho_shared[kernel_index * block_size
												+ threadIdx.x] *=
												intermediate_variables[(update_order
														+ 3) % order][index_3
														+ kernel_index];
										ho_shared[kernel_index * block_size
												+ threadIdx.x] *=
												intermediate_variables[(update_order
														+ 4) % order][index_4
														+ kernel_index];
										ho_shared[kernel_index * block_size
												+ threadIdx.x] *=
												intermediate_variables[(update_order
														+ 5) % order][index_5
														+ kernel_index];
										ho_shared[kernel_index * block_size
												+ threadIdx.x] *=
												intermediate_variables[(update_order
														+ 6) % order][index_6
														+ kernel_index];
										ho_shared[kernel_index * block_size
												+ threadIdx.x] *=
												intermediate_variables[(update_order
														+ 7) % order][index_7
														+ kernel_index];
										type_of_data gs_temp =
												parameter_b[(update_order + 8)
														% order][kernel_index
														* core_dimen + lane_id];
										gs_temp *= ho_shared[kernel_index
												* block_size + threadIdx.x];
										gs += gs_temp;

									}

									for (int order_index_8 = start_7;
											order_index_8 < end_7;
											order_index_8++) {

										int order_8 =
												idx_train_device[update_order][8][order_index_8];
										int index_8 = core_dimen * order_8;

										type_of_data parameter_a_temp =
												parameter_a[(update_order + 8)
														% order][index_8
														+ lane_id];

										type_of_data p_a_gs = parameter_a_temp
												* gs;

										int temp = core;
										while (temp != 1) {
											temp /= 2;
											p_a_gs += __shfl_down_sync(mask,
													p_a_gs, temp);
										}

										p_a_gs = __shfl_sync(mask, p_a_gs,
												(local_id % local) * core);

										p_a_gs -=
												value_train_device[update_order][order_index_8];

										for (int core_kernel_index = 0;
												core_kernel_index < core_kernel;
												core_kernel_index++) {
											atomicAdd(
													&b_sum[(order_index_8
															% sum_size)
															* core_kernel
															* core_dimen
															+ core_kernel_index
																	* core_dimen
															+ lane_id],
													p_a_gs

													* parameter_a_temp
															* ho_shared[core_kernel_index
																	* block_size
																	+ threadIdx.x]);
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

__global__ void Update_Parameter_B_CSF_Gradient_Order_10(const int order,
		const int update_order, const int core_kernel, const int core_dimen,
		int nnz, int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device, type_of_data **parameter_a,
		type_of_data **parameter_b, type_of_data *b_sum,
		type_of_data **intermediate_variables) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local = warp_size / core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data shared[];
	type_of_data *ho_shared = shared;

	for (int order_index_0 = ptr_train_device[update_order][0][0] + worker_id;
			order_index_0 < ptr_train_device[update_order][0][1];
			order_index_0 += workers) {

		int start_0 = ptr_train_device[update_order][1][order_index_0];
		int end_0 = ptr_train_device[update_order][1][order_index_0 + 1];

		int order_0 = idx_train_device[update_order][0][order_index_0];
		int index_0 = core_kernel * order_0;

		for (int order_index_1 = start_0 + local_id; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_kernel * order_1;

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int start_2 = ptr_train_device[update_order][3][order_index_2];
				int end_2 = ptr_train_device[update_order][3][order_index_2 + 1];
				int order_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_kernel * order_2;

				for (int order_index_3 = start_2; order_index_3 < end_2;
						order_index_3++) {

					int start_3 =
							ptr_train_device[update_order][4][order_index_3];
					int end_3 = ptr_train_device[update_order][4][order_index_3
							+ 1];
					int order_3 =
							idx_train_device[update_order][3][order_index_3];
					int index_3 = core_kernel * order_3;

					for (int order_index_4 = start_3; order_index_4 < end_3;
							order_index_4++) {

						int start_4 =
								ptr_train_device[update_order][5][order_index_4];
						int end_4 =
								ptr_train_device[update_order][5][order_index_4
										+ 1];
						int order_4 =
								idx_train_device[update_order][4][order_index_4];
						int index_4 = core_kernel * order_4;

						for (int order_index_5 = start_4; order_index_5 < end_4;
								order_index_5++) {

							int start_5 =
									ptr_train_device[update_order][6][order_index_5];
							int end_5 =
									ptr_train_device[update_order][6][order_index_5
											+ 1];
							int order_5 =
									idx_train_device[update_order][5][order_index_5];
							int index_5 = core_kernel * order_5;

							for (int order_index_6 = start_5;
									order_index_6 < end_5; order_index_6++) {

								int start_6 =
										ptr_train_device[update_order][7][order_index_6];
								int end_6 =
										ptr_train_device[update_order][7][order_index_6
												+ 1];
								int order_6 =
										idx_train_device[update_order][6][order_index_6];
								int index_6 = core_kernel * order_6;

								for (int order_index_7 = start_6;
										order_index_7 < end_6;
										order_index_7++) {

									int start_7 =
											ptr_train_device[update_order][8][order_index_7];
									int end_7 =
											ptr_train_device[update_order][8][order_index_7
													+ 1];
									int order_7 =
											idx_train_device[update_order][7][order_index_7];
									int index_7 = core_kernel * order_7;

									for (int order_index_8 = start_7;
											order_index_8 < end_7;
											order_index_8++) {

										int start_8 =
												ptr_train_device[update_order][9][order_index_8];
										int end_8 =
												ptr_train_device[update_order][9][order_index_8
														+ 1];
										int order_8 =
												idx_train_device[update_order][8][order_index_8];
										int index_8 = core_kernel * order_8;

										type_of_data gs = 0.0;

										for (int kernel_index = 0;
												kernel_index < core_kernel;
												kernel_index++) {

											ho_shared[kernel_index * block_size
													+ threadIdx.x] = 1.0f;
											ho_shared[kernel_index * block_size
													+ threadIdx.x] *=
													intermediate_variables[update_order][index_0
															+ kernel_index];
											ho_shared[kernel_index * block_size
													+ threadIdx.x] *=
													intermediate_variables[(update_order
															+ 1) % order][index_1
															+ kernel_index];
											ho_shared[kernel_index * block_size
													+ threadIdx.x] *=
													intermediate_variables[(update_order
															+ 2) % order][index_2
															+ kernel_index];
											ho_shared[kernel_index * block_size
													+ threadIdx.x] *=
													intermediate_variables[(update_order
															+ 3) % order][index_3
															+ kernel_index];
											ho_shared[kernel_index * block_size
													+ threadIdx.x] *=
													intermediate_variables[(update_order
															+ 4) % order][index_4
															+ kernel_index];
											ho_shared[kernel_index * block_size
													+ threadIdx.x] *=
													intermediate_variables[(update_order
															+ 5) % order][index_5
															+ kernel_index];
											ho_shared[kernel_index * block_size
													+ threadIdx.x] *=
													intermediate_variables[(update_order
															+ 6) % order][index_6
															+ kernel_index];
											ho_shared[kernel_index * block_size
													+ threadIdx.x] *=
													intermediate_variables[(update_order
															+ 7) % order][index_7
															+ kernel_index];
											ho_shared[kernel_index * block_size
													+ threadIdx.x] *=
													intermediate_variables[(update_order
															+ 8) % order][index_8
															+ kernel_index];
											type_of_data gs_temp =
													parameter_b[(update_order
															+ 9) % order][kernel_index
															* core_dimen
															+ lane_id];
											gs_temp *= ho_shared[kernel_index
													* block_size + threadIdx.x];
											gs += gs_temp;

										}

										for (int order_index_9 = start_8;
												order_index_9 < end_8;
												order_index_9++) {

											int order_9 =
													idx_train_device[update_order][9][order_index_9];
											int index_9 = core_dimen * order_9;

											type_of_data parameter_a_temp =
													parameter_a[(update_order
															+ 9) % order][index_9
															+ lane_id];

											type_of_data p_a_gs =
													parameter_a_temp * gs;

											int temp = core;
											while (temp != 1) {
												temp /= 2;
												p_a_gs += __shfl_down_sync(mask,
														p_a_gs, temp);
											}

											p_a_gs = __shfl_sync(mask, p_a_gs,
													(local_id % local) * core);

											p_a_gs -=
													value_train_device[update_order][order_index_9];

											for (int core_kernel_index = 0;
													core_kernel_index
															< core_kernel;
													core_kernel_index++) {
												atomicAdd(
														&b_sum[(order_index_9
																% sum_size)
																* core_kernel
																* core_dimen
																+ core_kernel_index
																		* core_dimen
																+ lane_id],
														p_a_gs
																* parameter_a_temp
																* ho_shared[core_kernel_index
																		* block_size
																		+ threadIdx.x]);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

__global__ void Parameter_B_Gradient_Sum(const int core_kernel,
		const int core_dimen, const int nnz,
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
		b_grad[core_kernel_index * core_dimen + lane_id] /= nnz;
	}
}

__global__ void Update_Parameter_B(const int update_order,
		const int core_kernel, const int core_dimen, type_of_data **parameter_b,
		type_of_data *b_grad, const type_of_data learn_rate_b,
		const type_of_data lambda_b) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int core_kernel_index = worker_id; core_kernel_index < core_kernel;
			core_kernel_index += workers) {

		parameter_b[update_order][core_kernel_index * core_kernel + lane_id] -=
				learn_rate_b
						* (b_grad[core_kernel_index * core_dimen + lane_id]
								+ lambda_b
										* parameter_b[update_order][core_kernel_index
												* core_kernel + lane_id]);
	}
}

void Update_Parameter_B_Batch(const int order, int *dimen,
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

		if (order == 3) {
			Update_Parameter_B_CSF_Gradient_Order_3 <<<grid_size,
			block_size, core_kernel * block_size* sizeof(type_of_data)>>>(order,
					update_order, core_kernel, core_dimen, nnz,
					ptr_train_len_device, idx_train_len_device,
					ptr_train_device, idx_train_device, value_train_device,
					parameter_a, parameter_b, b_sum, intermediate_variables);
			cudaDeviceSynchronize();
		} else if (order == 4) {
			Update_Parameter_B_CSF_Gradient_Order_4 <<<grid_size,
			block_size, core_kernel * block_size* sizeof(type_of_data)>>>(order,
					update_order, core_kernel, core_dimen, nnz,
					ptr_train_len_device, idx_train_len_device,
					ptr_train_device, idx_train_device, value_train_device,
					parameter_a, parameter_b, b_sum, intermediate_variables);
			cudaDeviceSynchronize();
		} else if (order == 5) {
			Update_Parameter_B_CSF_Gradient_Order_5 <<<grid_size,
			block_size, core_kernel * block_size* sizeof(type_of_data)>>>(order,
					update_order, core_kernel, core_dimen, nnz,
					ptr_train_len_device, idx_train_len_device,
					ptr_train_device, idx_train_device, value_train_device,
					parameter_a, parameter_b, b_sum, intermediate_variables);
			cudaDeviceSynchronize();
		} else if (order == 6) {
			Update_Parameter_B_CSF_Gradient_Order_6 <<<grid_size,
			block_size, core_kernel * block_size* sizeof(type_of_data)>>>(order,
					update_order, core_kernel, core_dimen, nnz,
					ptr_train_len_device, idx_train_len_device,
					ptr_train_device, idx_train_device, value_train_device,
					parameter_a, parameter_b, b_sum, intermediate_variables);
			cudaDeviceSynchronize();
		} else if (order == 7) {
			Update_Parameter_B_CSF_Gradient_Order_7 <<<grid_size,
			block_size, core_kernel * block_size* sizeof(type_of_data)>>>(order,
					update_order, core_kernel, core_dimen, nnz,
					ptr_train_len_device, idx_train_len_device,
					ptr_train_device, idx_train_device, value_train_device,
					parameter_a, parameter_b, b_sum, intermediate_variables);
			cudaDeviceSynchronize();
		} else if (order == 8) {
			Update_Parameter_B_CSF_Gradient_Order_8 <<<grid_size,
			block_size, core_kernel * block_size* sizeof(type_of_data)>>>(order,
					update_order, core_kernel, core_dimen, nnz,
					ptr_train_len_device, idx_train_len_device,
					ptr_train_device, idx_train_device, value_train_device,
					parameter_a, parameter_b, b_sum, intermediate_variables);
			cudaDeviceSynchronize();
		} else if (order == 9) {
			Update_Parameter_B_CSF_Gradient_Order_9 <<<grid_size,
			block_size, core_kernel * block_size* sizeof(type_of_data)>>>(order,
					update_order, core_kernel, core_dimen, nnz,
					ptr_train_len_device, idx_train_len_device,
					ptr_train_device, idx_train_device, value_train_device,
					parameter_a, parameter_b, b_sum, intermediate_variables);
			cudaDeviceSynchronize();
		} else if (order == 10) {
			Update_Parameter_B_CSF_Gradient_Order_10 <<<grid_size,
			block_size, core_kernel * block_size* sizeof(type_of_data)>>>(order,
					update_order, core_kernel, core_dimen, nnz,
					ptr_train_len_device, idx_train_len_device,
					ptr_train_device, idx_train_device, value_train_device,
					parameter_a, parameter_b, b_sum, intermediate_variables);
			cudaDeviceSynchronize();
		}

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

__global__ void RMSE_AND_MAE_CSF(const int order, const int update_order,
		const int core_kernel, const int core_dimen, int **ptr_train_len_device,
		int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device,
		type_of_data **value_train_device, type_of_data **parameter_a,
		type_of_data **parameter_b, type_of_data *rmse,
		type_of_data *mae) {

	extern __shared__ type_of_data shared[];

	type_of_data *qs_shared = shared;

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
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

		for (int kernel_index = 0; kernel_index < core_kernel; kernel_index++) {

			type_of_data temp_0 = parameter_a[0][index_0 + lane_id]
					* parameter_b[0][kernel_index * core_dimen + lane_id];

			int temp_temp_0 = core;
			while (temp_temp_0 != 1) {
				temp_temp_0 /= 2;
				temp_0 += __shfl_down_sync(mask, temp_0, temp_temp_0);
			}
			temp_0 = __shfl_sync(mask, temp_0, (local_id % local) * core);

			if (lane_id == 0) {
				qs_shared[0 * worker * core_kernel + local_id * core_kernel
						+ kernel_index] = temp_0;
			}
		}

		for (int order_index_1 = start_0; order_index_1 < end_0;
				order_index_1++) {

			int start_1 = ptr_train_device[update_order][2][order_index_1];
			int end_1 = ptr_train_device[update_order][2][order_index_1 + 1];
			int order_1 = idx_train_device[update_order][1][order_index_1];
			int index_1 = core_dimen * order_1;

			for (int kernel_index = 0; kernel_index < core_kernel;
					kernel_index++) {

				type_of_data temp_1 = parameter_a[1][index_1 + lane_id]
						* parameter_b[1][kernel_index * core_dimen + lane_id];

				int temp_temp_1 = core;
				while (temp_temp_1 != 1) {
					temp_temp_1 /= 2;
					temp_1 += __shfl_down_sync(mask, temp_1, temp_temp_1);
				}

				temp_1 = __shfl_sync(mask, temp_1, (local_id % local) * core);

				if (lane_id == 0) {
					qs_shared[1 * worker * core_kernel + local_id * core_kernel
							+ kernel_index] = temp_1;
				}

			}

			for (int order_index_2 = start_1; order_index_2 < end_1;
					order_index_2++) {

				int oder_2 = idx_train_device[update_order][2][order_index_2];
				int index_2 = core_dimen * oder_2;
				type_of_data val =
						value_train_device[update_order][order_index_2];

				for (int kernel_index = 0; kernel_index < core_kernel;
						kernel_index++) {
					type_of_data temp_2 =
							parameter_a[2][index_2 + lane_id]
									* parameter_b[2][kernel_index * core_dimen
											+ lane_id];

					int temp_temp_2 = core;
					while (temp_temp_2 != 1) {
						temp_temp_2 /= 2;
						temp_2 += __shfl_down_sync(mask, temp_2, temp_temp_2);
					}

					temp_2 = __shfl_sync(mask, temp_2,
							(local_id % local) * core);
					if (lane_id == 0) {
						val -= qs_shared[0 * worker * core_kernel
								+ local_id * core_kernel + kernel_index]
								* qs_shared[1 * worker * core_kernel
										+ local_id * core_kernel + kernel_index]
								* temp_2;
					}
				}

				if (lane_id == 0) {
					atomicAdd(&rmse[order_index_2 % error_size], val * val);
					atomicAdd(&mae[order_index_2 % error_size], abs(val));
				}
			}
		}
	}
}

void GET_RMSE_AND_MAE_CSF(const int order, const int core_kernel,
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

	RMSE_AND_MAE_CSF<<<grid_size, block_size, (order - 1) * block_size
	* core_kernel / core_dimen * sizeof(type_of_data)>>>(order, 0, core_kernel,
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

__global__ void RMSE_AND_MAE_COO(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, const type_of_data *value,
		int **index, type_of_data *rmse,
		type_of_data *mae) {

	int core = core_dimen;
	int worker = block_size / core;
	int local = warp_size / core;
	int lane_id = threadIdx.x % core;
	int local_wid = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_wid;
	int workers = worker * gridDim.x;

	extern __shared__ type_of_data shared[];

	type_of_data *gs_shared = shared;
	type_of_data *b_shared = (type_of_data*) &shared[order * block_size];

	for (int i = local_wid; i < order * core_kernel; i += worker) {
		b_shared[i * core_dimen + lane_id] = parameter_b[i / core_kernel][(i
				% core_kernel) * core_dimen + lane_id];
	}
	__syncthreads();

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		type_of_data p_a_gs = 0.0;
		for (int order_index = 0; order_index < order; order_index++) {
			type_of_data gs = 0.0;

			for (int core_kernel_index = 0; core_kernel_index < core_kernel;
					core_kernel_index++) {
				type_of_data gs_temp = b_shared[order_index * core_kernel
						* core_dimen + core_kernel_index * core_dimen + lane_id];

				for (int inner_order_index = 0; inner_order_index < order;
						inner_order_index++) {
					if (inner_order_index != order_index) {
						type_of_data temp =
								parameter_a[inner_order_index][index[inner_order_index][nnz_index]
										* core_dimen + lane_id]
										* b_shared[inner_order_index
												* core_kernel * core_dimen
												+ core_kernel_index * core_dimen
												+ lane_id];

						int temp_temp = core;
						while (temp_temp != 1) {
							temp_temp /= 2;
							temp += __shfl_down_sync(mask, temp, temp_temp);
						}

						temp = __shfl_sync(mask, temp,
								(local_wid % local) * core);

						gs_temp *= temp;

					}
				}
				gs += gs_temp;
			}
			gs_shared[order_index * block_size + threadIdx.x] = gs;
		}
		p_a_gs = parameter_a[0][index[0][nnz_index] * core_dimen + lane_id]
				* gs_shared[threadIdx.x];

		int temp = core;
		while (temp != 1) {
			temp /= 2;
			p_a_gs += __shfl_down_sync(mask, p_a_gs, temp);
		}

		p_a_gs = __shfl_sync(mask, p_a_gs, (local_wid % local) * core);

		p_a_gs -= value[nnz_index];

		if (lane_id == 0) {
			atomicAdd(&rmse[nnz_index % error_size], p_a_gs * p_a_gs);
			atomicAdd(&mae[nnz_index % error_size], abs(p_a_gs));
		}
	}

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

	RMSE_AND_MAE_COO<<<nnz / block_size + 1, block_size,
	(order * block_size + order * core_kernel * core_dimen)
	* sizeof(type_of_data)>>>(order, core_kernel, core_dimen, parameter_a,
			parameter_b, nnz, value, index, errors_rmse, errors_mae);
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
