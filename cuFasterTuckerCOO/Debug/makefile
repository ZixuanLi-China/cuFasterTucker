################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := cuFasterTuckerCOO
BUILD_ARTIFACT_EXTENSION :=
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: main-build

# Main-build Target
main-build: cuFasterTuckerCOO

# Tool invocations
cuFasterTuckerCOO: $(OBJS) $(USER_OBJS) makefile $(OPTIONAL_TOOL_DEPS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC linker'
	/usr/local/cuda-12.1/bin/nvcc --cudart=static -L/usr/local/cuda/lib64/ -ccbin g++ -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o "cuFasterTuckerCOO" $(OBJS) $(USER_OBJS) $(LIBS) -lcublas
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) cuFasterTuckerCOO
	-@echo ' '

.PHONY: all clean dependents main-build

-include ../makefile.targets
