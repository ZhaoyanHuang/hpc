static char help[] ="emplicit euler .\n\n";

#include <petscksp.h>
#include <petscmath.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>
#include <math.h>
#include <hdf5.h>

#define pi acos(-1)

int main(int argc,char **args)
{
  Vec            x, z, b, temp;       
  Mat            A;             
  PetscErrorCode ierr;           
  PetscInt       i, n=100, start=0, end=n, it=0, col[3], rstart,rend,nlocal,rank,index;
  PetscReal      p=1.0, c=1.0, k=1.0, alpha, beta, dx, f, dt=0.00001, t=0.0, u0=0.0;
  PetscScalar    zero = 0.0, value[3], data[3];
  PetscInt       restart = 0; 
  PetscViewer    h5; 

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;  
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);    
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);   
  ierr = PetscOptionsGetInt(NULL,NULL,"-restart",&restart,NULL);CHKERRQ(ierr);   


  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD, "n = %d\n", n);CHKERRQ(ierr); 

  dx=1.0/n;
  alpha = k/p/c;
  beta = alpha*dt/dx/dx;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"restart = %d\n",restart);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"dx = %f\n",dx);CHKERRQ(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"dt = %f\n",dt);CHKERRQ(ierr); 
  /* create vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&temp);CHKERRQ(ierr);

  ierr = VecSetSizes(x,PETSC_DECIDE,n+1);CHKERRQ(ierr); 
  ierr = VecSetSizes(temp, 3, PETSC_DECIDE);CHKERRQ(ierr);

  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(temp);CHKERRQ(ierr); 
  ierr = VecDuplicate(x,&z);CHKERRQ(ierr);  
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nlocal);CHKERRQ(ierr); 
  /* create matrix A */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr); 
  ierr = MatSetSizes(A,nlocal,nlocal,n+1,n+1);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(A);CHKERRQ(ierr); 
  ierr = MatSetUp(A);CHKERRQ(ierr); 

  /* give values */
  if (!rstart) 
  {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 1.0-2.0*beta; value[1] = beta; 
    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr); 
  }
  
  if (rend == n+1)
  {
    rend = n;
    i    = n; col[0] = n-1; col[1] = n; value[0] = beta; value[1] = 1.0-2.0*beta;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  value[0] = beta; value[1] = 1.0-2.0*beta; value[2] = beta;
  for (i=rstart; i<rend; i++) 
  {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr); 
  }
  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


 ierr = VecSet(z,zero);CHKERRQ(ierr);
    if(rank == 0)
    {
        for(int i=1; i<n; i++){ 
          u0 = exp(i*dx); 
          ierr = VecSetValues(z, 1, &i, &u0, INSERT_VALUES);CHKERRQ(ierr);
        }
    }
    ierr = VecAssemblyBegin(z);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(z);CHKERRQ(ierr);


  if(restart > 0)
  {
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"exp_petsc.h5", FILE_MODE_READ, &h5);CHKERRQ(ierr);    
    ierr = PetscObjectSetName((PetscObject) z, "exp_heat_z");CHKERRQ(ierr);   
    ierr = PetscObjectSetName((PetscObject) temp, "exp_heat_temp");CHKERRQ(ierr); 
    ierr = VecLoad(temp, h5);CHKERRQ(ierr);   
    ierr = VecLoad(z, h5);CHKERRQ(ierr);    
    ierr = PetscViewerDestroy(&h5);CHKERRQ(ierr);   
    
    index=0;   
    ierr = VecGetValues(temp,1,&index,&dx);CHKERRQ(ierr);    
    index += 1;    
    ierr = VecGetValues(temp,1,&index,&dt);CHKERRQ(ierr);    
    index += 1;  
    ierr = VecGetValues(temp,1,&index,&t);CHKERRQ(ierr);   
    index= 0;   
  }



  ierr = VecSet(b,zero);CHKERRQ(ierr); 
  if(rank == 0){
    for(int i = 1; i < n; i++){ 
      f = dt*sin(pi*i*dx);
      ierr = VecSetValues(b, 1, &i, &f, INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);


  while(PetscAbsReal(t)<2.0){ 
     t += dt; 
     /*x = Az*/
     ierr = MatMult(A,z,x);CHKERRQ(ierr);
     /*x = x+b*/
     ierr = VecAXPY(x,1.0,b);CHKERRQ(ierr);
     /* set vec value*/
     ierr = VecSetValues(x, 1, &start, &zero, INSERT_VALUES);CHKERRQ(ierr);
     ierr = VecSetValues(x, 1, &end, &zero, INSERT_VALUES);CHKERRQ(ierr);
     ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
     ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
     ierr = VecCopy(x,z);CHKERRQ(ierr);

     it += 1;
     if((it % 10) == 0)
     {
        data[0] = dx; data[1] = dt; data[2] = t;   
        ierr = VecSet(temp,zero);CHKERRQ(ierr);    
        for(index = 0; index < 3; index++)
        {   
          u0 = data[index];   
          ierr = VecSetValues(temp,1,&index,&u0,INSERT_VALUES);CHKERRQ(ierr);   
        }
        ierr = VecAssemblyBegin(temp);CHKERRQ(ierr);   
        ierr = VecAssemblyEnd(temp);CHKERRQ(ierr);  

        ierr = PetscViewerCreate(PETSC_COMM_WORLD,&h5);CHKERRQ(ierr);   
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"exp_petsc.h5", FILE_MODE_WRITE, &h5);CHKERRQ(ierr);    
        ierr = PetscObjectSetName((PetscObject) z, "exp_heat_z");CHKERRQ(ierr);   
        ierr = PetscObjectSetName((PetscObject) temp, "exp_heat_temp");CHKERRQ(ierr);   
        ierr = VecView(temp, h5);CHKERRQ(ierr);    
        ierr = VecView(z, h5);CHKERRQ(ierr);   
        ierr = PetscViewerDestroy(&h5);CHKERRQ(ierr); 
      }
  }

  ierr = VecView(z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    PetscViewer pv;
    PetscViewerCreate(PETSC_COMM_WORLD,&pv);
    PetscViewerASCIIOpen(PETSC_COMM_WORLD,"final.dat",&pv);
    VecView(z, pv);
    PetscViewerDestroy(&pv);

  ierr = VecDestroy(&temp);CHKERRQ(ierr);  
  ierr = VecDestroy(&x);CHKERRQ(ierr);  
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

// EOF
