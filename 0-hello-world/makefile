# gpu architecture
# Turing: this machine's Quadro T2000 is sm_75. Its compute_75 PTX still JITs
# forward on Ampere or newer, so this runs on both.
sm=75
GENCODE_SM   = -gencode arch=compute_$(sm),code=sm_$(sm)
GENCODE_COMP = -gencode arch=compute_$(sm),code=compute_$(sm)

all:
	nvcc ${GENCODE_SM} ${GENCODE_COMP} hello-world.cu -o hello-world

clean:
	rm -rf hello-world

