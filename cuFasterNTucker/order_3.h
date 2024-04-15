#include "parameter.h"

void Update_Parameter_A_3(const int order, int *dimen, const int core_kernel,
		const int core_dimen, int nnz_train, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a_device,
		type_of_data **parameter_b_device, const type_of_data lambda_a,
		type_of_data **intermediate_variables,
		type_of_data **a_grad_up,
		type_of_data **a_grad_down, type_of_data **a_grad_up_host_to_device,
		type_of_data **a_grad_down_host_to_device);

void Update_Parameter_B_Batch_3(const int order, int *dimen,
		const int core_kernel, const int core_dimen, int nnz,
		int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		type_of_data lambda_b,
		type_of_data **intermediate_variables);

