
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main()
{
	float dx, dt, T, L = 1.0, u[1000000], a[1000000], k = 1.0,rho = 1.0,c = 1.0;
	int i, n1, n2, n = 0, total = 1;
	
	/*The steps of x and t*/
	dx = 0.01;
	dt = 0.001;
	T = 1;
	n1 = (int)(L / dx) + 1;		
	n2 = (int)(T / dt);
	
	/*The resulting file*/
	FILE *F;
	F = fopen("data.txt", "w");
	
	/*The initial temperature at t=0*/
	for (i = 0; i < n1; i++) {   
		if (i == 0 || i == n1 - 1) 
			u[i] = 0;
		else u[i] = exp(i*dx);
		fprintf(F, "%8.4f", u[i]);
	}
	fprintf(F, "\n"); 

	/*Iterative computing*/
	while (n < n2) {    
		for (i = 0; i < n1; i++) {
			/*The temperature on the left and right boundaries is always 0*/
			if (i == 0 || i == n1-1) {
				u[i] = 0;
				u[i] = a[i];
			}     
			else {
				a[i] = u[i] + k * dt*(u[i + 1] - 2 * u[i] + u[i - 1]) / (dx*dx) + dt*sin(3.14*i*dx);  
				u[i] = a[i];
			}
			
			fprintf(F, "%8.4f ", u[i]);
			if (total%n1 == 0) {             /*Output data file  */
				fprintf(F, "\n");
			}
			total++;
			n++;
		}
	}
	fclose(F);
	return 0;
}
