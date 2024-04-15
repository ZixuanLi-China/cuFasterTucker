#include <sys/time.h>
#include "parameter.h"

inline double Seconds() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void Getting_Input(char *InputPath_train, char *InputPath_test, int order,
		int **dimen, int *nnz_train, int ***ptr_train_len, int ****ptr_train,
		int ***idx_train_len, int ****idx_train, type_of_data ***value_train,
		int *nnz_test, int ***index_test, type_of_data **value_test,
		double *data_norm);

void Parameter_Initialization(int order, int core_kernel, int core_length,
		int core_dimen, int *dimen, double data_norm,
		type_of_data ***parameter_a,
		type_of_data ***parameter_b);

void Cuda_Parameter_Initialization(int order, int core_kernel, int core_length,
		int core_dimen, int *dimen_host, int nnz_train,
		int **ptr_train_len_host, int ***ptr_train_host,
		int **idx_train_len_host, int ***idx_train_host,
		type_of_data **value_train_host, int ***ptr_train_len_device,
		int ****ptr_train_device, int ***idx_train_len_device,
		int ****idx_train_device, type_of_data ***value_train_device,
		int ***ptr_train_len_host_to_device,
		int ***idx_train_len_host_to_device, int ****ptr_train_host_to_device,
		int ****idx_train_host_to_device,
		type_of_data ***value_train_host_to_device, int nnz_test,
		int **index_test_host, int ***index_test_host_to_device,
		int ***index_test_device, type_of_data *value_test_host,
		type_of_data **value_test_device, type_of_data **parameter_a_host,
		type_of_data **parameter_b_host, type_of_data ***parameter_a_device,
		type_of_data ***parameter_b_device,
		type_of_data ***parameter_a_host_to_device,
		type_of_data ***parameter_b_host_to_device,
		type_of_data ***intermediate_variables_device,
		type_of_data ***intermediate_variables_host_to_device,
		type_of_data ***a_grad_up, type_of_data ***a_grad_down,
		type_of_data ***a_grad_up_host_to_device,
		type_of_data ***a_grad_down_host_to_device);

void Select_Best_Result(type_of_data *train_rmse, type_of_data *train_mae,
		type_of_data *test_rmse, type_of_data *test_mae,
		type_of_data *best_train_rmse, type_of_data *best_train_mae,
		type_of_data *best_test_rmse, type_of_data *best_test_mae);
