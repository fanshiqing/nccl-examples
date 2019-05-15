## Environment variables
#export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

## Build
echo "Start build..."
cd ./build
make clean
cmake -DNCCL_LIBRARY=/usr/local/nccl_2.3.5-2+cuda9.0_x86_64/lib/libnccl.so -DNCCL_INCLUDE_DIR=/usr/local/nccl_2.3.5-2+cuda9.0_x86_64/include ..
make example-1
cd ../
echo "Build success"

echo "Example 1: Single Process, Single Thread, Multiple Devices"
build/examples/example-1

#echo "Example 2: One Device Per Process Or Thread"
#mpirun -np 2 build/examples/example-2

#echo "Example 3: Multiple Devices Per Thread"
#mpirun -np 2 build/examples/example-3
