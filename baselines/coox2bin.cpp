// adapted from el2bin.cpp
#include <cstdio>
#include <fstream>
#include <random>
#include "GLib.hpp"
#include <cmath>
#include <cstring>

using namespace std;

int main(int argc, char ** argv)
{
	srand(time(NULL));
	int n = 0, u = 0, v = 0;
	long long m = 0;
	float w = 0.0;

    char coof_name[256];
    sprintf(coof_name, "%s%s", argv[1], "rows.tsv");
	printf("Reading file: %s!\n", coof_name);
	ifstream in_rows(coof_name);
    sprintf(coof_name, "%s%s", argv[1], "cols.tsv");
	printf("Reading file: %s!\n", coof_name);
	ifstream in_cols(coof_name);
    sprintf(coof_name, "%s%s", argv[1], "costs.tsv");
	printf("Reading file: %s!\n", coof_name);
	ifstream in_costs(coof_name);
	in_cols.ignore(10, '\n');
	in_costs.ignore(10, '\n');
	in_rows >> m;

    sprintf(coof_name, "%s%s", argv[1], "/filtered_ents.npy");
	ifstream in_ents(coof_name);
	in_ents >> n;
	in_ents.close();
	printf("%d %lld\n", n, m);

	vector<int> degree(n+1,0);
	vector<vector<int> > eList(n+1);
	vector<vector<float> > weight(n+1);
	vector<float> weightR(n+1,0);

	printf("Reading the graph!\n");

	for (long long i = 0; i < m; ++i){
        in_rows >> u;
        in_cols >> v;
        in_costs >> w;
        u++; v++;
		degree[v]++;
		eList[v].push_back(u);
		weight[v].push_back(1.0);
		weightR[u] += 1;
	}
	printf("Finished!\n");

	in_rows.close();
	in_cols.close();
	in_costs.close();

    vector<size_t> idx(n);

	FILE * pFile;
	pFile = fopen(argv[2],"wb");
	fwrite(&n, sizeof(int), 1, pFile);
	fwrite(&m, sizeof(long long), 1, pFile);

        for (int i = 0; i < n; ++i){
		idx[i] = i;
	}
	vector<int> inv_idx(n);
	for (int i = 0; i < n; ++i){
		inv_idx[idx[i]]	= i;
	}

	vector<int> iTmp(n);

	for (int i = 0; i < n; ++i){
		iTmp[i] = degree[idx[i]+1];
	}

	// Write node degrees
	fwrite(&iTmp[0], sizeof(int), n, pFile);

	for (int i = 1; i <= n; ++i){
		// Write neighbors
		for (unsigned int j = 0; j < eList[idx[i-1]+1].size(); ++j){
			iTmp[j] = inv_idx[eList[idx[i-1]+1][j]-1]+1;
		}
		fwrite(&iTmp[0], sizeof(int), eList[idx[i-1]+1].size(), pFile);
	}

	for (int i = 1; i <= n; ++i){
		// Write weights
                fwrite(&weight[idx[i-1] + 1][0], sizeof(float), weight[idx[i-1]+1].size(), pFile);
        }

	fclose(pFile);
	printf("Done!\n");
	return 1;
}
