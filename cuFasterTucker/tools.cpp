#include <fstream>
#include <cuda_runtime.h>
#include <string.h>
#include <math.h>
#include "parameter.h"

using namespace std;

type_of_data frand(type_of_data x, type_of_data y) {
	return ((y - x) * ((type_of_data) rand() / RAND_MAX)) + x;
}

void ReadLine_int(const char *filename, int *a) {
	string temp;
	int i = 0;
	fstream file;
	file.open(filename, ios::in);
	while (getline(file, temp)) {
		a[i] = atoi(temp.c_str());
		i++;
	}
	file.close();
}

void ReadLine_type_of_data(const char *filename, type_of_data *a) {
	string temp;
	int i = 0;
	fstream file;
	file.open(filename, ios::in);
	while (getline(file, temp)) {
		a[i] = atof(temp.c_str());
		i++;
	}
	file.close();
}

void Getting_Input(char *InputPath_train, char *InputPath_test, int order,
		int **dimen, int *nnz_train, int ***ptr_train_len, int ****ptr_train,
		int ***idx_train_len, int ****idx_train, type_of_data ***value_train,
		int *nnz_test, int ***index_test, type_of_data **value_test,
		double *data_norm) {

	*data_norm = 0.0;
	*dimen = (int*) malloc(sizeof(int) * order);
	for (int i = 0; i < order; i++) {
		(*dimen)[i] = 0;
	}

	*ptr_train_len = (int**) malloc(sizeof(int*) * order);
	*idx_train_len = (int**) malloc(sizeof(int*) * order);

	*ptr_train = (int***) malloc(sizeof(int**) * order);
	*idx_train = (int***) malloc(sizeof(int**) * order);

	for (int i = 0; i < order; i++) {
		(*ptr_train_len)[i] = (int*) malloc(sizeof(int) * order);
		(*idx_train_len)[i] = (int*) malloc(sizeof(int) * order);
		(*ptr_train)[i] = (int**) malloc(sizeof(int*) * order);
		(*idx_train)[i] = (int**) malloc(sizeof(int*) * order);
		for (int j = 0; j < order; j++) {
			(*ptr_train_len)[i][j] = 0;
			(*idx_train_len)[i][j] = 0;
		}
	}

	char tmp[1024];

	for (int i = 0; i < order; i++) {
		for (int j = 0; j < order; j++) {
			char *ptr_file_temp = new char[strlen(InputPath_train) + 30];
			char *idx_file_temp = new char[strlen(InputPath_train) + 30];
			sprintf(ptr_file_temp, "%strain_%d_%d_ptr.txt", InputPath_train, i,
					j);
			sprintf(idx_file_temp, "%strain_%d_%d_idx.txt", InputPath_train, i,
					j);

			FILE *ptr_file_count = fopen(ptr_file_temp, "r");
			FILE *idx_file_count = fopen(idx_file_temp, "r");
			while (fgets(tmp, 1024, ptr_file_count)) {
				(*ptr_train_len)[i][j]++;

			}
			while (fgets(tmp, 1024, idx_file_count)) {
				(*idx_train_len)[i][j]++;
			}
			fclose(ptr_file_count);
			fclose(idx_file_count);
		}
	}

	for (int i = 0; i < order; i++) {
		for (int j = 0; j < order; j++) {
			(*ptr_train)[i][j] = (int*) malloc(
					sizeof(int) * (*ptr_train_len)[i][j]);
			(*idx_train)[i][j] = (int*) malloc(
					sizeof(int) * (*idx_train_len)[i][j]);
			char *ptr_file_temp = new char[strlen(InputPath_train) + 20];
			char *idx_file_temp = new char[strlen(InputPath_train) + 20];
			sprintf(ptr_file_temp, "%strain_%d_%d_ptr.txt", InputPath_train, i,
					j);
			sprintf(idx_file_temp, "%strain_%d_%d_idx.txt", InputPath_train, i,
					j);
			ReadLine_int(ptr_file_temp, (*ptr_train)[i][j]);
			ReadLine_int(idx_file_temp, (*idx_train)[i][j]);
		}
	}

	for (int i = 0; i < order; i++) {
		int max = 0;
		for (int j = 0; j < (*idx_train_len)[i][0]; j++) {
			if (max < (*idx_train)[i][0][j]) {
				max = (*idx_train)[i][0][j];
			}
		}
		(*dimen)[i] = max + 1;

	}

	char *value_file_temp = new char[strlen(InputPath_train) + 20];
	sprintf(value_file_temp, "%strain_%d_val.txt", InputPath_train, 0);
	FILE *value_file_count = fopen(value_file_temp, "r");
	while (fgets(tmp, 1024, value_file_count)) {
		type_of_data temp = atof(tmp);
		(*data_norm) += temp * temp;
		(*nnz_train)++;
	}
	*data_norm = sqrt((*data_norm) / (*nnz_train));
	fclose(value_file_count);

	*value_train = (type_of_data**) malloc(sizeof(type_of_data*) * order);
	for (int i = 0; i < order; i++) {
		(*value_train)[i] = (type_of_data*) malloc(
				sizeof(type_of_data) * (*nnz_train));
		char *value_file = new char[strlen(InputPath_train) + 20];
		sprintf(value_file, "%strain_%d_val.txt", InputPath_train, i);
		ReadLine_type_of_data(value_file, (*value_train)[i]);
	}

	*index_test = (int**) malloc(sizeof(int*) * order);

	char *value_file_temp_test = new char[strlen(InputPath_test) + 20];
	sprintf(value_file_temp_test, "%stest_val.txt", InputPath_test);
	FILE *value_file_count_test = fopen(value_file_temp_test, "r");
	while (fgets(tmp, 1024, value_file_count_test)) {
		(*nnz_test)++;
	}
	fclose(value_file_count_test);

	for (int i = 0; i < order; i++) {
		(*index_test)[i] = (int*) malloc(sizeof(int) * (*nnz_test));
		char *index_file = new char[strlen(InputPath_test) + 20];
		sprintf(index_file, "%stest_index_%d.txt", InputPath_test, i);
		ReadLine_int(index_file, (*index_test)[i]);
	}

	*value_test = (type_of_data*) malloc(sizeof(type_of_data) * (*nnz_test));
	char *value_file = new char[strlen(InputPath_test) + 20];
	sprintf(value_file, "%stest_val.txt", InputPath_test);
	ReadLine_type_of_data(value_file, *value_test);

}

void Parameter_Initialization(int order, int core_kernel, int core_length,
		int core_dimen, int *dimen, double data_norm,
		type_of_data ***parameter_a, type_of_data ***parameter_b) {

	srand((unsigned) time(NULL));

	*parameter_a = (type_of_data**) malloc(sizeof(type_of_data*) * order);
	*parameter_b = (type_of_data**) malloc(sizeof(type_of_data*) * order);

	for (int i = 0; i < order; i++) {
		(*parameter_a)[i] = (type_of_data*) malloc(
				sizeof(type_of_data) * dimen[i] * core_dimen);
		(*parameter_b)[i] = (type_of_data*) malloc(
				sizeof(type_of_data) * core_dimen * core_kernel);
	}

	for (int i = 0; i < order; i++) {

		for (int j = 0; j < dimen[i] * core_dimen; j++) {
			(*parameter_a)[i][j] = pow(data_norm / core_length, 1.0 / order)
					* frand(0.5, 1.5);
		}

		for (int j = 0; j < core_kernel * core_dimen; j++) {
			(*parameter_b)[i][j] = pow(1.0 / core_kernel, 1.0 / order)
					* frand(0.5, 1.5);
		}

	}
}

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
		type_of_data ***intermediate_variables_host_to_device) {

	cudaMalloc((void**) &(*ptr_train_len_device), sizeof(int*) * order);
	cudaMalloc((void**) &(*idx_train_len_device), sizeof(int*) * order);

	*ptr_train_len_host_to_device = (int**) malloc(sizeof(int*) * order);
	*idx_train_len_host_to_device = (int**) malloc(sizeof(int*) * order);

	cudaMalloc((void**) &(*ptr_train_device), sizeof(int**) * order);
	cudaMalloc((void**) &(*idx_train_device), sizeof(int**) * order);
	cudaMalloc((void**) &(*value_train_device), sizeof(type_of_data*) * order);

	*ptr_train_host_to_device = (int***) malloc(sizeof(type_of_data**) * order);
	*idx_train_host_to_device = (int***) malloc(sizeof(type_of_data**) * order);
	*value_train_host_to_device = (type_of_data**) malloc(
			sizeof(type_of_data*) * order);

	for (int i = 0; i < order; i++) {

		int *temp_ptr_train_len;
		cudaMalloc((void**) &temp_ptr_train_len, sizeof(int) * order);
		(*ptr_train_len_host_to_device)[i] = temp_ptr_train_len;
		cudaMemcpy(temp_ptr_train_len, ptr_train_len_host[i],
				sizeof(int) * order, cudaMemcpyHostToDevice);

		int *temp_idx_train_len;
		cudaMalloc((void**) &temp_idx_train_len, sizeof(int) * order);
		(*idx_train_len_host_to_device)[i] = temp_idx_train_len;
		cudaMemcpy(temp_idx_train_len, idx_train_len_host[i],
				sizeof(int) * order, cudaMemcpyHostToDevice);

		(*ptr_train_host_to_device)[i] = (int**) malloc(sizeof(int*) * order);
		(*idx_train_host_to_device)[i] = (int**) malloc(sizeof(int*) * order);

		for (int j = 0; j < order; j++) {

			int *temp_temp_ptr_train;
			cudaMalloc((void**) &temp_temp_ptr_train,
					sizeof(int) * ptr_train_len_host[i][j]);
			(*ptr_train_host_to_device)[i][j] = temp_temp_ptr_train;
			cudaMemcpy(temp_temp_ptr_train, ptr_train_host[i][j],
					sizeof(int) * ptr_train_len_host[i][j],
					cudaMemcpyHostToDevice);

			int *temp_temp_idx_train;
			cudaMalloc((void**) &temp_temp_idx_train,
					sizeof(int) * idx_train_len_host[i][j]);
			(*idx_train_host_to_device)[i][j] = temp_temp_idx_train;
			cudaMemcpy(temp_temp_idx_train, idx_train_host[i][j],
					sizeof(int) * idx_train_len_host[i][j],
					cudaMemcpyHostToDevice);

		}

		int **temp_ptr_train;
		cudaMalloc((void**) &temp_ptr_train, sizeof(int*) * order);
		cudaMemcpy(temp_ptr_train, (*ptr_train_host_to_device)[i],
				sizeof(int*) * order, cudaMemcpyHostToDevice);
		(*ptr_train_host_to_device)[i] = temp_ptr_train;

		int **temp_idx_train;
		cudaMalloc((void**) &temp_idx_train, sizeof(int*) * order);
		cudaMemcpy(temp_idx_train, (*idx_train_host_to_device)[i],
				sizeof(int*) * order, cudaMemcpyHostToDevice);
		(*idx_train_host_to_device)[i] = temp_idx_train;

		type_of_data *temp_val_train;
		cudaMalloc((void**) &temp_val_train, sizeof(type_of_data) * nnz_train);
		(*value_train_host_to_device)[i] = temp_val_train;
		cudaMemcpy(temp_val_train, value_train_host[i],
				sizeof(type_of_data) * nnz_train, cudaMemcpyHostToDevice);

	}

	cudaMemcpy(*ptr_train_len_device, *ptr_train_len_host_to_device,
			sizeof(int*) * order, cudaMemcpyHostToDevice);
	cudaMemcpy(*idx_train_len_device, *idx_train_len_host_to_device,
			sizeof(int*) * order, cudaMemcpyHostToDevice);

	cudaMemcpy((*ptr_train_device), (*ptr_train_host_to_device),
			sizeof(int**) * order, cudaMemcpyHostToDevice);
	cudaMemcpy((*idx_train_device), (*idx_train_host_to_device),
			sizeof(int**) * order, cudaMemcpyHostToDevice);
	cudaMemcpy((*value_train_device), (*value_train_host_to_device),
			sizeof(type_of_data*) * order, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &(*index_test_device), sizeof(int*) * order);
	*index_test_host_to_device = (int**) malloc(sizeof(int*) * order);
	cudaMalloc((void**) &(*value_test_device), sizeof(type_of_data) * nnz_test);

	for (int i = 0; i < order; i++) {
		int *temp_index_test;
		cudaMalloc((void**) &temp_index_test, sizeof(int) * nnz_test);
		(*index_test_host_to_device)[i] = temp_index_test;
		cudaMemcpy(temp_index_test, index_test_host[i], sizeof(int) * nnz_test,
				cudaMemcpyHostToDevice);
	}

	cudaMemcpy(*index_test_device, *index_test_host_to_device,
			sizeof(int*) * order, cudaMemcpyHostToDevice);
	cudaMemcpy(*value_test_device, value_test_host,
			sizeof(type_of_data) * nnz_test, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &(*parameter_a_device), sizeof(type_of_data*) * order);
	cudaMalloc((void**) &(*parameter_b_device), sizeof(type_of_data*) * order);

	*parameter_a_host_to_device = (type_of_data**) malloc(
			sizeof(type_of_data*) * order);
	*parameter_b_host_to_device = (type_of_data**) malloc(
			sizeof(type_of_data*) * order);

	for (int i = 0; i < order; i++) {

		type_of_data *temp_a;
		cudaMalloc((void**) &temp_a,
				sizeof(type_of_data) * dimen_host[i] * core_dimen);
		(*parameter_a_host_to_device)[i] = temp_a;
		cudaMemcpy(temp_a, parameter_a_host[i],
				sizeof(type_of_data) * dimen_host[i] * core_dimen,
				cudaMemcpyHostToDevice);

		type_of_data *temp_b;
		cudaMalloc((void**) &temp_b,
				sizeof(type_of_data) * core_kernel * core_dimen);
		(*parameter_b_host_to_device)[i] = temp_b;
		cudaMemcpy(temp_b, parameter_b_host[i],
				sizeof(type_of_data) * core_kernel * core_dimen,
				cudaMemcpyHostToDevice);

	}

	cudaMemcpy(*parameter_a_device, *parameter_a_host_to_device,
			sizeof(type_of_data*) * order, cudaMemcpyHostToDevice);
	cudaMemcpy(*parameter_b_device, *parameter_b_host_to_device,
			sizeof(type_of_data*) * order, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &(*intermediate_variables_device),
			sizeof(type_of_data*) * order);
	*intermediate_variables_host_to_device = (type_of_data**) malloc(
			sizeof(type_of_data*) * order);

	for (int i = 0; i < order; i++) {

		type_of_data *temp_intermediate_variables;
		cudaMalloc((void**) &temp_intermediate_variables,
				sizeof(type_of_data) * dimen_host[i] * core_kernel);
		(*intermediate_variables_host_to_device)[i] =
				temp_intermediate_variables;

	}

	cudaMemcpy(*intermediate_variables_device,
			*intermediate_variables_host_to_device,
			sizeof(type_of_data*) * order, cudaMemcpyHostToDevice);

}

void Select_Best_Result(type_of_data *train_rmse, type_of_data *train_mae,
		type_of_data *test_rmse, type_of_data *test_mae,
		type_of_data *best_train_rmse, type_of_data *best_train_mae,
		type_of_data *best_test_rmse, type_of_data *best_test_mae) {

	if (*train_rmse < *best_train_rmse) {
		*best_train_rmse = *train_rmse;
	}
	if (*train_mae < *best_train_mae) {
		*best_train_mae = *train_mae;
	}
	if (*test_rmse < *best_test_rmse) {
		*best_test_rmse = *test_rmse;
	}
	if (*test_mae < *best_test_mae) {
		*best_test_mae = *test_mae;
	}
}
