#include <cuda_runtime.h>
#include <curand_kernel.h>

#define type_of_data float

void Intermediate_Variables_Initialization(const int order, int *dimen,
		const int core_kernel, const int core_dimen,
		type_of_data **parameter_a_device, type_of_data **parameter_b_device,
		type_of_data **intermediate_variables_device);

void Update_Parameter_A(const int order, int *dimen, const int core_kernel,
		const int core_dimen, int nnz_train, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a_device,
		type_of_data **parameter_b_device, const type_of_data learn_rate_a,
		const type_of_data lambda_a, type_of_data **intermediate_variables);

void Update_Parameter_B_Batch(const int order, int *dimen,
		const int core_kernel, const int core_dimen, int nnz,
		int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		type_of_data learn_rate_b, type_of_data lambda_b,
		type_of_data **intermediate_variables);

void GET_RMSE_AND_MAE_CSF(const int order, const int core_kernel,
		const int core_dimen, const int nnz, int **ptr_train_len_device,
		int **idx_train_len_device, int ***ptr_train_device,
		int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		type_of_data *rmse,
		type_of_data *mae);

void GET_RMSE_AND_MAE_COO(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, type_of_data *value,
		int **index, type_of_data *rmse, type_of_data *mae);
