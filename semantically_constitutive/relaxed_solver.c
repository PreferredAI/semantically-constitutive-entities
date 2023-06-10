#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <libgen.h>
#include "gurobi_c.h"

#define opencol(j)         j
#define transportcol(COOIdx)  nFacilities + COOIdx
#define MAXSTR             256

int* load_coox(char filepath[]) {

  char buf[MAXSTR];
  int matrixDims[2];
  int i = 0, j = 0;

  FILE *inFile = fopen(filepath, "r");

  // test for input file not existing.
  if ( inFile == NULL ) {
    puts("Error! Could not open file\n");
    exit(EXIT_FAILURE); // must include stdlib.h
  }

  // first line should contain the size of the matrix
  fgets( buf, sizeof(buf), inFile );
  char *token = strtok(buf, "\t");
  i = 0;
  while( token ) {
    if ( i > 1 ) {  // make sure xFile is 2-dimensional
      puts("X file dims is greater than 2\n");
      exit(EXIT_FAILURE);
    }
    matrixDims[i] = atoi(token);
    printf("matrixDims[%i] == %i\n", i, matrixDims[i]);
    token = strtok(NULL, "\t");
    i += 1;
  }

  int *y_hat = calloc( (matrixDims[1] + 1), sizeof(int) );
  y_hat[0] = matrixDims[1];

  while ( fgets( buf, sizeof(buf), inFile ) != NULL ) {  // read line from inFile in coo format: (i, j, x_ij)
    char *token = strtok(buf, "\t");
    j = 0;
    while( token ) {
      if ( j == 1 ) {
        int j_idx = atoi(token);
        y_hat[j_idx + 1] = 1;
        break;
      } else {
        j++;
        token = strtok(NULL, "\t");
      }
    }
  }
  fclose(inFile);

  return y_hat;
}

double* load_coo_file_double(char filepath[]) {

  char buf[MAXSTR];
  printf("reading from: %s\n", filepath);
  FILE *inFile = fopen(filepath, "r");

  // test for inFile file not existing.
  if ( inFile == NULL ) {
    puts("Error! Could not open file\n");
    exit(EXIT_FAILURE); // must include stdlib.h
  }

  // first line should contain the number of elements to read
  fgets( buf, sizeof(buf), inFile );
  int i = atoi(buf);
  double *result = malloc( sizeof(double) * (i + 1) );
  result[0] = (double) i;

  i = 1;
  while ( fgets( buf, sizeof(buf), inFile ) != NULL ) {  // read chunk from inFile
    result[i] = atof(buf);
    i += 1;
  }
  fclose(inFile);

  return result + 1;

}

int* load_coo_file_int(char filepath[]) {

  char buf[MAXSTR];
  printf("reading from: %s\n", filepath);
  FILE *inFile = fopen(filepath, "r");

  // test for inFile file not existing.
  if ( inFile == NULL ) {
    puts("Error! Could not open file\n");
    exit(EXIT_FAILURE); // must include stdlib.h
  }

  // first line should contain the number of elements to read
  fgets( buf, sizeof(buf), inFile );
  int i = atoi(buf);
  int *result = malloc( sizeof(int) * (i + 1) );
  result[0] = i;

  i = 1;
  while ( fgets( buf, sizeof(buf), inFile ) != NULL ) {  // read chunk from inFile
    result[i] = atoi(buf);
    i += 1;
  }
  fclose(inFile);

  return result + 1;

}

int main(int argc, char *argv[]) {
  setbuf(stdout, NULL);

  int i, j;

  /* Gurobi variables */
  GRBenv   *env   = NULL;
  GRBmodel *model = NULL;
  int       error = 0;
  int idx, col, numnz;
  char     vname[MAXSTR];
  int      *cind = NULL;
  double   *cval = NULL;
  char     *cname = NULL;
  double   sol, obj;
  float ghostCost;
  int **validMatrix;
  int COOIdx;

  char *cijDir = NULL;
  char *gurobiXFile = NULL;
  char *outDir = NULL;
  float numClusters = 1000.0;
  float timeLimit = 86400.0;  // stop gurobi MIP optimization after this many seconds; default 86400s = 24h
  int numGhosts = 0;  // compensate for locations with <k facilities; reassign weights back to actual recon
  int numFree = 0;    // but if location is itself a facility, we use free facilities to compensate instead

  while ( (i = getopt( argc, argv, "c:x:k:g:o:t:" ) ) != -1) {
    switch(i) {
      case 'c':
        cijDir = optarg;
        break;
      case 'o':
        outDir = optarg;
        break;
      case 'x':
        gurobiXFile = optarg;
        break;
      case 'k':
        numClusters = atof(optarg);
        break;
      case 'g':
        numGhosts = numFree = atoi(optarg);
        break;
      case 't':
        timeLimit = atof(optarg);
        break;
      default:
        printf("found invalid option: %c\n", i);
        exit(1);
    }
  }
  printf("run_args\t\n\tc: %s\n\to: %s\n\tx: %s\n\tg: %d\n\tk: %f\n\tt: %f\n", cijDir, outDir, gurobiXFile, numGhosts, numClusters, timeLimit);

  printf("loading y_hat from gurobi relaxed X: %s\n", gurobiXFile);
  int* yHatPtr = load_coox(gurobiXFile);
  const int nFacilitiesX = yHatPtr[0];
  int* yHat = yHatPtr + 1;

  j = 0;
  for ( i = 0; i < nFacilitiesX; i++ ) {
    j += yHat[i];
  }
  printf("\tfound %i facilities in y_hat\n", j);

  printf("loading c_ij from %s!\n", cijDir);

  sprintf(vname, "%s/rows.tsv", cijDir);
  int *allRowsCOO = load_coo_file_int(vname);
  sprintf(vname, "%s/cols.tsv", cijDir);
  int *allColsCOO = load_coo_file_int(vname);
  sprintf(vname, "%s/costs.tsv", cijDir);
  double *allCostsCOO = load_coo_file_double(vname);
  vname[0] = '\0';  // empty vname for later use

  const int nLocations = nFacilitiesX;
  const int nFacilitiesReal = nFacilitiesX;
  const int nFacilities = nFacilitiesReal + numGhosts + numFree;
  printf("shape: (%i, %i)\n", nLocations, nFacilities);

  // remove cost c_ij if j not in yHat
  ghostCost = 0.0;
  int ghostNumnz = nLocations * (numGhosts + numFree);
  int allNumnz = allRowsCOO[-1];
  numnz = 0;
  for ( i=0; i < allNumnz; i++ ) {
    numnz += yHat[allColsCOO[i]];
  }
  int *rowsCOO = malloc( sizeof(int) * (numnz + ghostNumnz) );
  int *colsCOO = malloc( sizeof(int) * (numnz + ghostNumnz) );
  double *costsCOO = malloc( sizeof(double) * (numnz + ghostNumnz) );
  j = 0;
  for ( i=0; i < allNumnz; i++ ) {
    if ( yHat[allColsCOO[i]] ) {
      rowsCOO[j] = allRowsCOO[i];
      colsCOO[j] = allColsCOO[i];
      costsCOO[j++] = allCostsCOO[i];
      ghostCost = ghostCost > allCostsCOO[i] ? ghostCost : allCostsCOO[i];
    }
  }
  ghostCost = 5 * ghostCost;
  printf("ghostCost (5x max cost): %f\n", ghostCost);
  for ( i = 0; i < nLocations; i++ ) {
    for ( j = nFacilitiesReal; j < nFacilities; j++ ) {
      rowsCOO[numnz] = i;
      colsCOO[numnz] = j;
      costsCOO[numnz++] = ( j - nFacilities ) < numGhosts ? ghostCost : 0;
    }
  }
  // free all*COO files; don't need them anymore
  free(allRowsCOO - 1);
  free(allColsCOO - 1);
  free(allCostsCOO - 1);
  printf("REGUROBI: reduced C numnz from %i to %i\n", allNumnz, numnz);

  printf("creating valid matrix: %d x %d for %d non-zeros\n", nFacilities, nFacilities, numnz);
  validMatrix = (int **) malloc( sizeof(int *) * nFacilities );
  for( i = 0; i < nFacilities; i++ ) {
    validMatrix[i] = calloc( nFacilities, sizeof(int) );
  }
  for ( i = 0; i < numnz; i++ ) {
    validMatrix[rowsCOO[i]][colsCOO[i]] = i + 1;
  }
  #define GetCOOIdx(i, j) ( validMatrix[i][j] > 0 ? validMatrix[i][j] - 1 : -1 )

  printf("allocations done!\n");

  error = GRBemptyenv(&env);
  if (error) goto QUIT;

  error = GRBstartenv(env);
  if (error) goto QUIT;

  /* Create initial model */
  error = GRBnewmodel(env, &model, "facility", nFacilities + numnz,
                      NULL, NULL, NULL, NULL, NULL);  // refer to GRBnewmodel documentation for variable inits
  if (error) goto QUIT;

  printf("SETTING GUROBI TIME LIMIT: %f seconds\n", timeLimit);
  error = GRBsetdblparam(GRBgetenv(model), GRB_DBL_PAR_TIMELIMIT, timeLimit);  // measured in seconds; 86400 = 24h
  if (error) goto QUIT;
  printf("model init done!\n");

  /* Initialize decision variables for plant open variables */
  for ( j = 0; j < nFacilities; j++ ) {
    col = opencol(j);
    error = GRBsetcharattrelement(model, "VType", col, GRB_BINARY);
    if (error) goto QUIT;
    if ( j > nFacilitiesReal ) {
      error = GRBsetdblattrelement(model, GRB_DBL_ATTR_LB, col, 1.0);
      if (error) goto QUIT;
    }
    error = GRBsetdblattrelement(model, GRB_DBL_ATTR_UB, col, 1.0);
    if (error) goto QUIT;
    sprintf(vname, "Y_%i", j); // Y_j in model
    error = GRBsetstrattrelement(model, "VarName", col, vname);
    if (error) goto QUIT;
  }
  printf("Y_j allocation done!\n");

  /* Initialize decision variables for transportation decision variables:
     how much to transport from a plant p to a warehouse w */
  for ( COOIdx = 0; COOIdx < numnz; COOIdx++ ) {
    col = transportcol(COOIdx);
    error = GRBsetcharattrelement(model, "VType", col, GRB_BINARY);
    if (error) goto QUIT;
    error = GRBsetdblattrelement(model, "Obj", col, costsCOO[COOIdx]);
    if (error) goto QUIT;
    error = GRBsetdblattrelement(model, GRB_DBL_ATTR_UB, col, 1.0);
    if (error) goto QUIT;
    sprintf(vname, "X_%i_%i", rowsCOO[COOIdx], colsCOO[COOIdx]);
    error = GRBsetstrattrelement(model, "VarName", col, vname);
    if (error) goto QUIT;
  }

  printf("X_ij allocation done!\n");

  /* The objective is to minimize the total fixed and variable costs */
  error = GRBsetintattr(model, "ModelSense", GRB_MINIMIZE);
  if (error) goto QUIT;

  cind = malloc( sizeof(int) * numnz );
  if (!cind) goto QUIT;
  cval = malloc( sizeof(double) * numnz );
  if (!cval) goto QUIT;
  cname = calloc(MAXSTR, sizeof(char));
  if (!cname) goto QUIT;

  // Constraint 1: Each location i must be assigned to exactly K facilities
  for ( i = 0; i < nLocations; ++i ) {
    numnz = 0;
    idx = 0;
    sprintf(cname, "X_%i_sum_j", i);
    for ( j = 0; j < nFacilities; j++ ) {
      COOIdx = GetCOOIdx(i, j);
      if ( COOIdx <= -1 )  {
        continue;
      }
      numnz += 1;
      cind[idx] = transportcol(COOIdx);
      cval[idx++] = 1.0;
    }
    error = GRBaddconstr(model, numnz, cind, cval, GRB_EQUAL,
                         numGhosts, cname);
  }
  printf("constraint 1 done!\n");

  // Constraint 2: Facilities can only provide as much as they are open
  printf("constraint 2 starting for %i facilities!\n", nLocations);
  for ( i = 0; i < nLocations; ++i ) {
    for ( j = 0; j < nFacilities; ++j ) {
      COOIdx = GetCOOIdx(i, j);
      if ( COOIdx <= -1 )  {
        continue;
      }
      cind[0] = opencol(j);
      cval[0] = 1.0;
      cind[1] = transportcol(COOIdx);
      cval[1] = -1.0;
      sprintf(cname, "Y_%i_GEQ_X_%i_%i", j, i, j);  // Y_j >= X_ij
      error = GRBaddconstr(model, 2, cind, cval, GRB_GREATER_EQUAL,
                           0.0, cname);
    }
  }
  printf("constraint 2 done!\n");

  // Constraint 3: Supply must sum to numClusters
  sprintf(cname, "sum_Y_j");
  for ( j = 0; j < nFacilitiesReal; j++ ) {
    cind[j] = opencol(j);
    cval[j] = 1.0;
  }
  error = GRBaddconstr(model, nFacilitiesReal, cind, cval, GRB_EQUAL,
                       numClusters, cname);
  if (error) goto QUIT;
  printf("constraint 3 done!\n");

  // Constraint 4 (FREE): Only allow free if location itself is a facility
  for ( j = 0; j < numFree; j++ ) {
    for ( i = 0; i < nLocations; ++i ) {
      COOIdx = GetCOOIdx(i, nFacilitiesReal + numGhosts + j);
      if ( COOIdx <= -1 )  {
        continue;
      }
      cind[0] = opencol(i);
      cval[0] = 1.0;
      cind[1] = transportcol(COOIdx);
      cval[1] = -1.0;
      sprintf(cname, "Y_%i_GEQ_X_%i_%i", i, rowsCOO[COOIdx], colsCOO[COOIdx]);  // Y_i >= X_ij
      error = GRBaddconstr(model, 2, cind, cval, GRB_GREATER_EQUAL,
                           0.0, cname);
    }
  }
  if (error) goto QUIT;
  printf("constraint 4 done!\n");

  // free COO files, all done with them!
  free(rowsCOO);
  free(colsCOO);
  free(costsCOO);
  free(cind);
  free(cval);
  free(cname);

  char fileName[256];
  fileName[0] = '\0';

  error = GRBsetintparam( GRBgetenv( model ), GRB_INT_PAR_THREADS, 4 );
  if (error) goto QUIT;
  error = GRBsetintparam(GRBgetenv(model),
                         GRB_INT_PAR_METHOD,
                         GRB_METHOD_DETERMINISTIC_CONCURRENT);
  if (error) goto QUIT;

  /* Solve */
  error = GRBoptimize(model);
  if (error) goto QUIT;

  /* Print solution */
  error = GRBgetdblattr(model, "ObjVal", &obj);
  if (error) goto QUIT;
  printf("\nTOTAL COSTS: %f\n", obj);

  // write X matrix to file
  const char cooFormat[] = "\n%i\t%i\t%f";
  fileName[0] = '\0';
  sprintf(fileName, "%s/%s-semcon-%i-%i.cooX", outDir, basename(gurobiXFile), (int) numClusters, (int) numGhosts);
  printf("Writing X: %s\n", fileName);
  FILE *outFile = fopen(fileName, "w");
  fprintf(outFile, "%i\t%i", nLocations, nFacilities);
  for ( i = 0; i < nLocations; ++i ) {
    for ( j = 0; j < nFacilities; ++j ) {
      COOIdx = GetCOOIdx(i, j);
      if ( COOIdx <= -1 )  {
        continue;
      } else {
        error = GRBgetdblattrelement(model, "X", nFacilities + COOIdx, &sol);
        if (error) goto QUIT;
        if ( sol > 0.0001 ) {
          fprintf(outFile, cooFormat, i, j, sol);
        }
      }
    }
  }
  fclose(outFile);

QUIT:

  /* Error reporting */

  if (error)
  {
    printf("ERROR (%i): %s\n", error, GRBgeterrormsg(env));
    exit(1);
  }

  /* Free model */

  GRBfreemodel(model);

  /* Free environment */

  GRBfreeenv(env);

  puts("\nall done!");
  return 0;
}
