#include <stdio.h>
#include <stdlib.h>

int main(){
  int A[100],B[100],C[100];
  for (int i=0, m=0; i<100; i++)
    {
      A[i] = i;
      B[i] = 100 - i;
      m ++;
      C[i] = m;
    }
  int j=0;
  while (j<100){
    j++;
    C[j] = A[j]+B[j];
  }
  for (int k=0; k<10; k++)
    {
      printf("C[%d]=%d\t",k,C[k]);
    }
  printf("\n");
  return 0;
}
