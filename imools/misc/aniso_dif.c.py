

Vector Anisotropic_diffusion(AnisoDiff *AD){
  
  int nr;
  char dummy[80];
  void *pointer[2];
  /*Vector v[4]; needed as parameter for cg-solver
    v[0]=AD->evec[0];  v[1]=AD->evec[1];
    v[2]=AD->alpha[0]; v[3]=AD->alpha[1];*/
  
  pointer[0]=(void*)AD;
  pointer[1]=(void*)&AD->sigma;

  /* Loop over time steps */
  for (nr=0;nr<AD->steps;nr++){

  
 
    /* Copy image (=result) ->conv */
    memcpy(AD->conv,AD->result,AD->size*sizeof(DTYPE));

    /* Convolving of result*/
    if(AD->sigma!=0) { 
      if (INFO>1)
	printf("   Convolving image with sigma = %f\n",AD->sigma);  
      setupRightSide(AD,AD->rhs,AD->result);
      getConvDiagonal(AD->diagonal,AD,AD->sigma); 
     
     
      AD->conv=PCG(convProduct, diagPre,
		   AD->rhs,AD->conv,AD->diagonal,
		   AD->cg_eps,AD->cg_steps,AD->size,pointer); 
    } 
    /*writePGM("conv.pgm", AD->conv, AD->width, AD->height,1,VFLIP)*/
   
    
    /* Calculation of structure tensor*/
    ComputeADTensor(AD,nr);
    
    /*Solving the PDE*/  

    if (INFO>1)
      printf("   Solving AD\n");fflush(stdout);
    if(AD->tau!=0) {
      setupRightSide(AD,AD->rhs,AD->result);
      AD->result=CG(ADProduct,AD->rhs,AD->result,
		    AD->cg_eps,AD->cg_steps,AD->size,AD);
    } 
    
    if (AD->write_smoothed_image){
      sprintf(dummy,"%s%s.out%03d.%s",
	      AD->data_dir,AD->filename_smoothed_image,nr+1,AD->format_smoothed_image);
      if (compare_ending(AD->format_smoothed_image,"fits"))
	write_fits_image(dummy,AD->result,FLOAT_IMG,AD->width,AD->height);
      else if (compare_ending(AD->format_smoothed_image,"pgm"))
	writePGM(dummy,AD->result,AD->width,AD->height,2,VFLIP);
    }
       
    /*sprintf(dummy,"%s.out%03d.pgm",P->tempfile,nr+1);
      writePGM(dummy,AD->result,P->width,P->height,2,VFLIP);*/ 
  }
  return AD->result;
}


void ComputeADTensor(AnisoDiff *AD,int step){
  
  /* multiple use of vectors (saves memory)*/
  /* 0 AD->conv
     1 AD->alpha[0]  <->   Tensor 1
     2 AD->alpha[1]  <->   Tensor 2
     3 AD->evec[0]   <->   Tensor 3
     4 AD->evec[1]   <->   Rhs for convolving tensor
     5 AD->coherence    
     6 eval1
     7 eval2  
     8 AD->force 
     9 AD->result
    10 AD->diagonal   <->diagonal
*/
  void *pointer[2];
  Vector source,tens[3],rhs,eval[2];
  int i;

  /* Calculate entries of structure tensor */
  source=AD->conv;
  tens[0]=AD->alpha[0];
  tens[1]=AD->alpha[1];
  tens[2]=AD->evec[0];
  eval[0]=newVector(AD->size);
  eval[1]=newVector(AD->size);
  /*coherence=newVector(AD->size);*/
  rhs=AD->evec[1];


  ComputeStructTensor(AD,source,tens);    

  /* Convolve tensor*/
  pointer[0]=AD;
  pointer[1]=&AD->rho;

  if(AD->rho!=0){
    if (INFO>1)
      printf("   Convolving of Diffmatrix (rho = %f )\n",AD->rho);fflush(stdout);
    getConvDiagonal(AD->diagonal,AD,AD->rho);
    for (i=0;i<3;i++){
      /*printf("   %d.\n ",i);fflush(stdout);*/
      setupRightSide(AD,rhs,tens[i]);     
      tens[i]=PCG(convProduct, diagPre,rhs,tens[i],AD->diagonal,
		  AD->cg_eps,AD->cg_steps,AD->size,pointer);
    }   
  }

 
  
  if (INFO>1)
    printf("   Central values\n");fflush(stdout);
  /* Calculate values in cells*/
  for (i=0;i<3;i++)
    Calc_centralvalues(AD,tens[i],rhs); 
 
#ifdef WRITE_INFO
  sprintf(dummy,"%sad.tens1.%d.pgm",AD->data_dir,step);
  writePGM(dummy,tens[0],AD->width,AD->height,2,VFLIP); 
  sprintf(dummy,"%sad.tens2.%d.pgm",P->data_dir,step);
  writePGM(dummy,tens[1],AD->width,AD->height,2,VFLIP);
  sprintf(dummy,"%sad.tens3.%d.pgm",P->data_dir,step);
  writePGM(dummy,tens[2],AD->width,AD->height,2,VFLIP);
#endif

  /* Calculate eigenvalues, eigenvectors and local coherence*/
  if (INFO>1)
    printf("   Eigenvalues\n");
  Calc_eigen(AD,tens,AD->evec,eval,AD->coherence);
  modify_eigenvalues_ad(AD,AD->alpha,eval,AD->coherence,AD->gKval,AD->tau);

#ifdef WRITE_INFO
  sprintf(dummy,"%sad.eval1.%d.pgm",AD->data_dir,step);
  write_rgb(dummy,eval[0],AD->width,AD->height,0,0,2,VFLIP); 
  sprintf(dummy,"%sad.eval2.%d.pgm",AD->data_dir,step);
  write_rgb(dummy,eval[1],AD->width,AD->height,0,0,2,VFLIP); 
  sprintf(dummy,"%sad.coh.%d.pgm",AD->data_dir,step); 
  write_rgb(dummy,AD->coherence,AD->width,AD->height,0,0,2,VFLIP);
  sprintf(dummy,"%sad.eval1.bw.%d.pgm",AD->data_dir,step); 
  writePGM(dummy,eval[0],AD->width,AD->height,2,VFLIP); 
  sprintf(dummy,"%sad.alpha1.%d.pgm",AD->data_dir,step); 
  write_rgb(dummy,AD->alpha[0],AD->width,AD->height,0,0,2,VFLIP);
  sprintf(dummy,"%sad.alpha2.%d.pgm",AD->data_dir,step); 
  write_rgb(dummy,AD->alpha[1],AD->width,AD->height,0,0,2,VFLIP); 

  sprintf(dummy,"%sad.evec1.%d.pgm",AD->data_dir,step); 
  write_rgb(dummy,AD->evec[0],AD->width,AD->height,0,0,2,VFLIP);
  sprintf(dummy,"%sad.evec2.%d.pgm",AD->data_dir,step); 
  write_rgb(dummy,AD->evec[1],AD->width,AD->height,0,0,2,VFLIP); 
#endif
 
  free(eval[0]);
  free(eval[1]);
}


void ComputeStructTensor(AnisoDiff *AD,Vector source, Vector tens[3]){

  DTYPE dux,duy;
  int x,y,x0,x1,y0,y1,k;

  for (x=0;x<AD->width;x++)
    for(y=0;y<AD->height;y++){ 

      x0=x-1;
      x1=x+1;
      y0=y-1;
      y1=y+1;

      /* calculate grad(u) =udx,udy */
      if (x0<1 || x1>=AD->width) 
	dux=0;
      else
	dux=(source[x1+AD->map[y]]-source[x0+AD->map[y]])/2.;
      if (y0<1 || y1>=AD->height) 
	duy=0;
      else
	duy=(source[x+AD->map[y1]]-source[x+AD->map[y0]])/2.;
      k=x+AD->map[y];
      tens[0][k]=dux*dux;
      tens[1][k]=dux*duy; 
      tens[2][k]=duy*duy;


      /* Calculate extensions on right and upper boundary to fill
	 vectors d[i] of size <size> 

      if (x1==AD->width-1 && y!=0){             
	l=x1+AD->map[y];
	duy=(source[x1+AD->map[y1]]-source[x1+AD->map[y0]])/2.;
	tens[0][l]=0.;
	tens[1][l]=0.;
	tens[2][l]=duy*duy;
      }                                  
      else if(y1==AD->height-1 && x!=0 ){   
	l=x+AD->map[y1];
	dux=(source[x1+AD->map[y1]]-source[x0+AD->map[y1]])/2.; 
	tens[0][l]=dux*dux;
	tens[1][l]=0.;
	tens[2][l]=0.;
      } */
    }
}



void Calc_centralvalues(AnisoDiff *AD,Vector vec,Vector temp){

  /* Center value in cell is average of value at corners */
  int x,y,k,edge;
  clearVector(temp,AD->size);
 
  for (x=0;x<AD->width-1;x++)
    for(y=0;y<AD->height-1;y++){ 
      k=x+AD->map[y];
      temp[k]=vec[k];
      for(edge=1;edge<4;edge++){
	temp[k]+=vec[k+AD->nmap[edge]]; 
      }
      temp[k]*=.25;
    }
  memcpy(vec,temp,AD->size*sizeof(DTYPE)); 
}



void Calc_eigen(AnisoDiff *AD,Vector tens[3], Vector evec[2], Vector eval[2],Vector coh) {

  DTYPE loceval[2],locevec[2],coherence;
  DTYPE loctens[3];
  DTYPE temp;
  int x,y,k;

  for (x=0;x<AD->width;x++)
    for(y=0;y<AD->height;y++){ 
      k=x+AD->map[y];  

      loctens[0]=tens[0][k];
      loctens[1]=tens[1][k];
      loctens[2]=tens[2][k];

      /* Eigenvectors*/ 
      if (loctens[1]==0){
	if (loctens[2]>loctens[0]){
	  locevec[0]=0.;
	  locevec[1]=1.;
	  loceval[0]=loctens[2];
	  loceval[1]=loctens[0];
	  coherence=loceval[0]-loceval[1];
	}
	else if(loctens[0]>loctens[2]) {
	  locevec[0]=1.;
	  locevec[1]=0.;
	  loceval[0]=loctens[0];
	  loceval[1]=loctens[2];
	  coherence=loceval[0]-loceval[1];
	}
	else {
	  locevec[0]=1.;
	  locevec[1]=1.;
	  loceval[0]=loctens[0];
	  loceval[1]=loctens[2];
	  coherence=loceval[0]-loceval[1];
	}
      }
      else{ 
	temp=loctens[2]-loctens[0];
	
	coherence=sqrt(temp*temp+4*loctens[1]*loctens[1]);
	locevec[0]=2*loctens[1];
	locevec[1]=temp+coherence;
	temp=sqrt(locevec[0]*locevec[0]+locevec[1]*locevec[1]);
	if (temp==0){
	  locevec[0]=1.;
	  locevec[1]=1.;
	  loceval[0]=1.;
	  loceval[1]=1.;
	  coherence=0.;
	}
	else{
	  locevec[0]/=temp;
	  locevec[1]/=temp;
	  
	  /* Eigenvalues */
	  loceval[0]=.5*(loctens[0]+loctens[2]+coherence);
	  loceval[1]=.5*(loctens[0]+loctens[2]-coherence);
	}
      }
      evec[0][k]=locevec[0]; 
      evec[1][k]=locevec[1];
      eval[0][k]=loceval[0];
      eval[1][k]=loceval[1];
      coh[k]=coherence;
    }   
}

void modify_eigenvalues_ad(AnisoDiff *AD,Vector alpha[2],Vector eval[2],Vector coherence, DTYPE gKval,DTYPE tau){
  int x,y,k;

  /*Modify eigenvalues */

   for (x=0;x<AD->width;x++)
    for(y=0;y<AD->height;y++){ 
      k=x+AD->map[y];  
      alpha[0][k]=tau*GVALUE(eval[0][k]);/*coherence or eval[0]*/
      alpha[1][k]=tau;
    }
}



