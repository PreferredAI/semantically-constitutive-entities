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

int* load_filtered_ents(char filepath[]) {

  char buf[MAXSTR];
  printf("reading from: %s\n", filepath);
  FILE *inFile = fopen(filepath, "r");

  // test for inFile file not existing.
  if ( inFile == NULL ) {
    puts("Error! Could not open file\n");
    exit(EXIT_FAILURE); // must include stdlib.h
  }

  // first line contains the maximum number of elements to read
  fgets( buf, sizeof(buf), inFile );
  int i = atoi(buf);
  int *result = malloc( sizeof(int) * (i + 1) );
  // we set all elements to -1 to indicate an invalid entity
  int j = 0;
  for ( j = 0; j < (i+1); j++ ) {
    result[j] = -1;
  }
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
  char *outDir = NULL;
  float numClusters = 1000.0;
  float timeLimit = 86400.0;  // stop gurobi MIP optimization after this many seconds; default 86400s = 24h
  int numGhosts = 0;  // compensate for locations with <k facilities; reassign weights back to actual recon
  int numFree = 0;    // but if location is itself a facility, we use free facilities to compensate instead

  while ( (i = getopt( argc, argv, "c:g:k:o:t:" ) ) != -1) {
    switch(i) {
      case 'c':
        cijDir = optarg;
        break;
      case 'o':
        outDir = optarg;
        break;
      case 'g':
        numGhosts = numFree = atoi(optarg);
        break;
      case 'k':
        numClusters = atof(optarg);
        break;
      case 't':
        timeLimit = atof(optarg);
        break;
    }
  }
  printf("run_args\t\n\tc: %s\n\to: %s\n\tg: %i\n\tk: %f\n\tt: %f\n", cijDir, outDir, numGhosts, numClusters, timeLimit);

  printf("loading c_ij from %s!\n", cijDir);

  sprintf(vname, "%s/rows.tsv", cijDir);
  int *rowsCOO = load_coo_file_int(vname);
  sprintf(vname, "%s/cols.tsv", cijDir);
  int *colsCOO = load_coo_file_int(vname);
  sprintf(vname, "%s/costs.tsv", cijDir);
  double *costsCOO = load_coo_file_double(vname);
  vname[0] = '\0';  // empty vname for later use

  numnz = *(rowsCOO - 1);  // equivalently, rowsCOO[-1]

  sprintf(vname, "%s/filtered_ents.npy", cijDir);
  // filteredEnts is the list of (valid) locations for this level of filtering
  // filteredEnts[-1] gives the total number of ents, and -1 indicates an invalid ent
  int *filteredEnts = load_filtered_ents(vname);
  const int nLocations = filteredEnts[-1];
  const int nFacilitiesReal = filteredEnts[-1];
  const int nFacilities = nFacilitiesReal + numGhosts + numFree;

  printf("shape: (%i, %i)\n", nLocations, nFacilities);

  if ( numGhosts ) {
    ghostCost = 0.0;
    for ( i = 0; i < numnz; i++ ) {
        ghostCost = ghostCost > costsCOO[i] ? ghostCost : costsCOO[i];
    }
    ghostCost = 5 * ghostCost;
    printf("ghostCost (5x max cost): %f\n", ghostCost);
  }

  // We want to expand all COOs to include ghost / free facilities
  idx = nLocations * (numGhosts + numFree);  // check number of new X variables required
  if ( idx > 0 ) {
    void* realloc_ptr;
    realloc_ptr = realloc(rowsCOO - 1, sizeof(int) * ( numnz + idx + 1 ) );
    if (realloc_ptr == NULL) goto QUIT; else rowsCOO = (int*)(realloc_ptr) + 1;
    realloc_ptr = realloc(colsCOO - 1, sizeof(int) * ( numnz + idx + 1 ) );
    if (realloc_ptr == NULL) goto QUIT; else colsCOO = (int*)(realloc_ptr) + 1;
    realloc_ptr = realloc(costsCOO - 1, sizeof(double) * ( numnz + idx + 1 ) );
    if (realloc_ptr == NULL) goto QUIT; else costsCOO = (double*)(realloc_ptr) + 1;
    for ( i = 0; i < nLocations; i++ ) {
      for ( j = nFacilitiesReal; j < nFacilities; j++ ) {
        rowsCOO[numnz] = i;
        colsCOO[numnz] = j;
        costsCOO[numnz++] = ( j - nFacilities ) < numGhosts ? ghostCost : 0;
      }
    }
  }

  printf("creating valid matrix: %d x %d for %d non-zeros\n", nFacilities, nFacilities, numnz);
  validMatrix = (int **) malloc( sizeof(int *) * nFacilities );
  for ( i = 0; i < nFacilities; i++ ) {
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
    if ( j > nFacilitiesReal ) {
      error = GRBsetdblattrelement(model, GRB_DBL_ATTR_LB, col, 1.0);
      if (error) goto QUIT;
    }
    error = GRBsetdblattrelement(model, GRB_DBL_ATTR_UB, col, 1.0);
    if (error) goto QUIT;
    sprintf(vname, "Y_%i", j);
    error = GRBsetstrattrelement(model, "VarName", col, vname);
    if (error) goto QUIT;
  }
  printf("Y_j allocation done!\n");

  /* Initialize decision variables for transportation decision variables:
     how much to transport from a plant p to a warehouse w */
  for ( COOIdx = 0; COOIdx < numnz; COOIdx++ ) {
    col = transportcol(COOIdx);
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

  // Constraint 3: Supply must sum to numClusters (+ numGhosts)
  sprintf(cname, "sum_Y_j");
  for ( j = 0; j < nFacilitiesReal; j++ ) {
    cind[j] = opencol(j);
    cval[j] = 1.0;
  }
  error = GRBaddconstr(model, nFacilitiesReal, cind, cval, GRB_LESS_EQUAL,
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
  free(rowsCOO - 1);
  free(colsCOO - 1);
  free(costsCOO - 1);
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

  const char cooFormat[] = "\n%i\t%i\t%f";
  fileName[0] = '\0';
  sprintf(fileName, "%s/%s-partial-%i-%i.cooX", outDir, basename(cijDir), (int) numClusters, numGhosts);
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
