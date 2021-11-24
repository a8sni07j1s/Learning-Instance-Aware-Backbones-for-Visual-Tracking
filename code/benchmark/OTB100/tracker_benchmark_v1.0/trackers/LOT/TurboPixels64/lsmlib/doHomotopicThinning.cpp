/*
 * File:        computeExtensionFields2d.c
 * Copyright:   (c) 2005-2006 Kevin T. Chu
 * Revision:    $Revision: 1.14 $
 * Modified:    $Date: 2006/09/18 16:17:02 $
 * Description: MATLAB MEX-file for using the fast marching method to
 *              compute extension fields for 2d level set functions
 */

/*===========================================================================
 *
 * computeExtensionFields2d() computes a distance from an 
 * arbitrary level set function using the Fast Marching Method.
 * 
 * Usage: [distance_function, extension_fields] = ...
 *        computeExtensionFields2d(phi, source_fields, dX, ...
 *                                 mask, ...
 *                                 spatial_derivative_order)
 *
 * Arguments:
 * - phi:                       level set function to use in 
 *                                computing distance function
 * - source_fields:             field variables that are to
 *                                be extended off of the zero
 *                                level set
 * - dX:                        array containing the grid spacing
 *                                in each coordinate direction
 * - mask:                      mask for domain of problem;
 *                                grid points outside of the domain
 *                                of the problem should be set to a
 *                                negative value
 *                                (default = [])
 * - spatial_derivative_order:  order of discretization for 
 *                                spatial derivatives
 *                                (default = 5)
 *
 * Return values:
 * - distance_function:         distance function
 * - extension_fields:          extension fields
 *
 * NOTES:
 * - All data arrays are assumed to be in the order generated by the
 *   MATLAB meshgrid() function.  That is, data corresponding to the
 *   point (x_i,y_j) is stored at index (j,i).
 *
 *===========================================================================*/

#include "mex.h"
#include "lsm_fast_marching_method.h" 

/* Input Arguments */
#define PHI	                   (prhs[0])
#define BACKGROUND                 (prhs[1])

/* Output Arguments */
#define SKELETON          (plhs[0])


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* field data */
  double *phi;
  double *thinned_img;
  double *background;
  mxLogical *skeleton;
  mxArray *thinned_img_array;
  
  /* grid data */
  const int *grid_dims = mxGetDimensions(PHI);

  /* Check for proper number of arguments */
  if (nrhs != 2 ) {
    mexErrMsgTxt(
        "Wrong number of input arguments (2 required;)");
  }

  /* Assign pointers for phi and extension field data */
  phi = mxGetPr(PHI);
  background = mxGetPr(BACKGROUND);
  
  /* Create distance function and extension field data */
  thinned_img_array = mxCreateDoubleMatrix(grid_dims[0], grid_dims[1], mxREAL);
  thinned_img = mxGetPr(thinned_img_array);

  /* Carry out FMM calculation */
  doHomotopicThinning(
      thinned_img,
      phi,
      background,
      (int*) grid_dims);

  SKELETON = mxCreateLogicalMatrix(grid_dims[0], grid_dims[1]);
  skeleton = mxGetLogicals(SKELETON);
  
  for (int j = 0; j < grid_dims[1]; j++)
  {
    for (int i = 0; i < grid_dims[0]; i++)
    {
      skeleton[j*grid_dims[0] + i] = thinned_img[j*grid_dims[0] + i] >= 0;
    }
  }
  
  mxDestroyArray(thinned_img_array);
  
  return;
}
