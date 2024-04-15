#include "parameter.h"

__global__ void Calculate_Intermediate_Variables(const int order,
		const int core_kernel, const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int update_order,
		const int update_order_dimen,
		type_of_data **intermediate_variables);

void Intermediate_Variables_Initialization(const int order, int *dimen,
		const int core_kernel, const int core_dimen,
		type_of_data **parameter_a_device, type_of_data **parameter_b_device,
		type_of_data **intermediate_variables_device);

__global__ void Update_Parameter_A(const int update_order, const int dimen,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data **a_grad_up,
		type_of_data **a_grad_down);

void Update_Parameter_A(const int order, int *dimen, const int core_kernel,
		const int core_dimen, int nnz_train, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a_device,
		type_of_data **parameter_b_device, const type_of_data lambda_a,
		type_of_data **intermediate_variables,
		type_of_data **a_grad_up,
		type_of_data **a_grad_down, type_of_data **a_grad_up_host_to_device,
		type_of_data **a_grad_down_host_to_device);

__global__ void Parameter_B_Gradient_Sum_Up(const int core_kernel,
		const int core_dimen,
		type_of_data *b_sum, type_of_data *b_grad);

__global__ void Parameter_B_Gradient_Sum_Down(const int core_kernel,
		const int core_dimen, const int nnz,
		type_of_data *b_sum, type_of_data *b_grad, type_of_data **parameter_b,
		const type_of_data lambda_b, const int update_order);

__global__ void Update_Parameter_B(const int update_order,
		const int core_kernel, const int core_dimen, type_of_data **parameter_b,
		type_of_data *b_grad_up, type_of_data *b_grad_down);

void Update_Parameter_B_Batch(const int order, int *dimen,
		const int core_kernel, const int core_dimen, int nnz,
		int **ptr_train_len_device, int **idx_train_len_device,
		int ***ptr_train_device, int ***idx_train_device,
		type_of_data **value_train_device,
		type_of_data **parameter_a, type_of_data **parameter_b,
		type_of_data lambda_b,
		type_of_data **intermediate_variables);

void GET_RMSE_AND_MAE_CSF_3(const int order, const int core_kernel,
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
