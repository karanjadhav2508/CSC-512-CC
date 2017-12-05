#include <stdio.h>

__global__ void child()
{
	int x = 0;
}

__global__ void parent()
{
	int x = 2;
	child<<<1,x>>>();
}

void func() {}

int main(void)
{
	int a = 0;
	parent<<<1,1>>>();
	func();
}
