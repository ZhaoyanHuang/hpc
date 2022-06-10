static char help[] = "impilict Euler.\n\n";

#include <petscksp.h>
#include <petscmath.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>
#include <math.h>

#define pi acos(-1) 

int main(int argc,char **args)
{
  Vec            x, b, u, t;        
  Mat            A;                
  KSP            ksp;
  PC             pc;
  PetscErrorCode ierr;             
  PetscInt       i, n=1000, start=0, end=n, col[3], rstart,rend,nlocal,rank, it=0, index; 
  PetscReal      c=1.0, k=1.0, rho=1.0, f, alpha, beta, dx, dt=0.0001, T=0.0, u0=0.0; 
  PetscScalar    zero = 0.0, value[3], data[3];
  PetscInt      restart = 0;
  PetscViewer    h5;


  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
   /* read from command line */
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(NULL,NULL,"-restart",&restart,NULL);CHKERRQ(ierr); 

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr); 
  end=n; 

  dx=1.0/n;
  alpha = k/rho/c;
  beta = alpha*dt/dx/dx;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"dx = %f\n",dx);CHKERRQ(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"dt = %f\n",dt);CHKERRQ(ierr);

  /* create vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&t);CHKERRQ(ierr); 

  ierr = VecSetSizes(x,PETSC_DECIDE,n+1);CHKERRQ(ierr);
  ierr = VecSetSizes(t, 3, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(t);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr); 
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr); 
  ierr = VecGetLocalSize(x,&nlocal);CHKERRQ(ierr);  

  ierr = VecSet(b,zero);CHKERRQ(ierr);  
      if(rank == 0)
      {
          for(int i=1; i<n; i++){ 
            u0 = exp(i*dx);  
            ierr = VecSetValues(b, 1, &i, &u0, INSERT_VALUES);CHKERRQ(ierr);
          }
      }
      ierr = VecAssemblyBegin(b);CHKERRQ(ierr); 
      ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  if(restart > 0)
   {   
      ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"imp_petsc.h5", FILE_MODE_READ, &h5);CHKERRQ(ierr);    
      ierr = PetscObjectSetName((PetscObject) b, "imp_heat_b");CHKERRQ(ierr);    
      ierr = PetscObjectSetName((PetscObject) t, "imp_heat_t");CHKERRQ(ierr);
      ierr = VecLoad(t, h5);CHKERRQ(ierr);    
      ierr = VecLoad(b, h5);CHKERRQ(ierr);    
      ierr = PetscViewerDestroy(&h5);CHKERRQ(ierr);  

      index=0;    
      ierr = VecGetValues(t,1,&index,&dx);CHKERRQ(ierr);    
      index += 1;    
      ierr = VecGetValues(t,1,&index,&dt);CHKERRQ(ierr);   
      index += 1;   
      ierr = VecGetValues(t,1,&index,&T);CHKERRQ(ierr);   
      index= 0;   
    }


  ierr = VecSet(u,zero);CHKERRQ(ierr); 
  if(rank == 0){
    for(int i = 1; i < n; i++){  
      f = dt*sin(pi*i*dx); 
      ierr = VecSetValues(u, 1, &i, &f, INSERT_VALUES);CHKERRQ(ierr); 
    }
  }

  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);  
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr); 
  ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  
  /* create matrix A */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,nlocal,nlocal,n+1,n+1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /*restart*/
  if (!rstart)
  {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 1.0+2.0*beta; value[1] = -beta;
    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  
  if (rend == n+1)
  {
    rend = n;
    i    = n; col[0] = n-1; col[1] = n; value[0] = -beta; value[1] = 1.0+2.0*beta;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  value[0] = -beta; value[1] = 1.0+2.0*beta; value[2] = -beta;
  for (i=rstart; i<rend; i++)
  {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*KSP solver*/
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);    
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);   
  /*error*/
  ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);   

  /*calculate*/
  while(PetscAbsReal(T)<2.0){  
     /*time step*/
     T += dt;   
     it += 1;
     ierr = VecAXPY(b,1.0,u);CHKERRQ(ierr);    
     ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);   

     ierr = VecSetValues(x, 1, &start, &zero, INSERT_VALUES);CHKERRQ(ierr); 
     ierr = VecSetValues(x, 1, &end, &zero, INSERT_VALUES);CHKERRQ(ierr);
     ierr = VecAssemblyBegin(x);CHKERRQ(ierr); 
     ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
     ierr = VecCopy(x,b);CHKERRQ(ierr);  
    
        if((it % 10)==0)
        {    

          data[0] = dx; data[1] = dt; data[2] = T;    
          ierr = VecSet(t,zero);CHKERRQ(ierr);   
          for(index=0;index<3;index++){    
            u0 = data[index];    
            ierr = VecSetValues(t,1,&index,&u0,INSERT_VALUES);CHKERRQ(ierr);    
          }
          ierr = VecAssemblyBegin(t);CHKERRQ(ierr);   
          ierr = VecAssemblyEnd(t);CHKERRQ(ierr);    

          ierr = PetscViewerCreate(PETSC_COMM_WORLD,&h5);CHKERRQ(ierr);   
          ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"imp_petsc.h5", FILE_MODE_WRITE, &h5);CHKERRQ(ierr);   
          ierr = PetscObjectSetName((PetscObject) b, "imp_heat_b");CHKERRQ(ierr);   
          ierr = PetscObjectSetName((PetscObject) t, "imp_heat_t");CHKERRQ(ierr);    
          ierr = VecView(t, h5);CHKERRQ(ierr);   
          ierr = VecView(b, h5);CHKERRQ(ierr);   
          ierr = PetscViewerDestroy(&h5);CHKERRQ(ierr);    
        }

  }

  ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
 
  ierr = VecDestroy(&t);CHKERRQ(ierr);ierr = VecDestroy(&x);CHKERRQ(ierr);  
  ierr = VecDestroy(&u);CHKERRQ(ierr);ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize(); 
  return ierr;
}

// EOF
