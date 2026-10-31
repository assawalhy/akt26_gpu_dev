CUDADIR ?= /usr/local/cuda

# gpu architecture
# Turing: this machine's Quadro T2000 is sm_75. Its compute_75 PTX still JITs
# forward on Ampere or newer, so this runs on both.
sm=75
GENCODE_SM   = -gencode arch=compute_$(sm),code=sm_$(sm)
GENCODE_COMP = -gencode arch=compute_$(sm),code=compute_$(sm)

LIB=-L${CUDADIR}/lib64 -lcublas

all:
	nvcc ${GENCODE_SM} ${GENCODE_COMP} gemm.cu -o gemm ${LIB}

clean:
	rm -rf gemm

