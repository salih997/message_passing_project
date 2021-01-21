#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <cmath>            
#include <vector>
#include <utility>
#include <bits/stdc++.h> 

using namespace std;

int main(int argc, char* argv[])
{
    int rank; // rank of the current processor
    int size; // total number of processors
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // gets the rank of the current processor
    MPI_Comm_size(MPI_COMM_WORLD, &size); // gets the total number of processors

    // ****************************************** //

    int P;
    int N,A,M,T;
    FILE *cin = fopen(argv[1], "r");
    fscanf(cin,"%d", &P);   // reads number of total processors from argument     P-1 =>slave
    fscanf(cin,"%d", &N);   // reads number of instance - number of lines -
    fscanf(cin,"%d", &A);   // reads number of features  - 1 line has A+1 value (for class label)-
    fscanf(cin,"%d", &M);   // reads number of iterations
    fscanf(cin,"%d", &T);   // reads resulting number of features - top T features are selected-

    int selected2 [T];
    vector <int>selected;
    int finalize [P*T];
    
    long double arr[N*(A+1)*P/(P-1)]; // storage array of master - needs to store N line and each line has (A+1) entry (but we need to fill first N*(A+1)/(P-1) entries)-
    long double pref[(N/(P-1))*(A+1)]; // storage array of slaves - a slave take N/(P-1) instance and each instance have (A+1) entry -

    // If it's master processor, reads from input file
    if(rank==0){
        long double num=0.0;
        int j=0, i=N*(A+1)/(P-1);
        for(;j<N*(A+1)/(P-1);j++)
            arr[j]=0.0;
        while(fscanf(cin, "%Lf", &num)==1){
            arr[i]=num;
            i++;
        }
        fclose(cin);
    }

    // sends data from root array arr to pref array on each processor
    MPI_Scatter(arr,N*(A+1)/(P-1),MPI_LONG_DOUBLE,pref,N*(A+1)/(P-1),MPI_LONG_DOUBLE,0,MPI_COMM_WORLD);

        // N*(A+1)/(P-1) one slave has this much entries - N/(P-1) line each line has (A+1) entry -

        if(rank!= 0){                       // slaves work here
            int myInstNum = N/(P-1);       // this frequently used value is instance of one slave


            long double instances[myInstNum][A+1];
            long double distances[myInstNum];

            long double weights [A] = {};

            for(int i=0; i<myInstNum; i++){                                 // this loop seperate pref to two dimension matrix to ease the operations
                for(int j=0; j<A+1; j++){
                    instances [i][j] = pref[(i*(A+1))+j];
                }
            }
            
            int target_type;
            int target, hit, mis;
            long double hit_dist=1.79769e+308, mis_dist=1.79769e+308;
            for(int i=0; i<M; i++){                                             
                    /// instances [i][]    ==>>   target instance
                    /// instance  [i][A]   ==>>   target_type
                target_type = instances[i][A];                        
                hit_dist=INT32_MAX, mis_dist=INT32_MAX;

                for(int j=0; j<myInstNum; j++){                 
                    long double tempdist = 0;
                    
                    for(int k=0; k<A; k++){
                        tempdist = tempdist + fabs(instances[i][k] - instances[j][k]);
                    }
                    distances[j] = tempdist;
                }

                for(int j=0; j<myInstNum; j++){
                    if (j==i) continue;                 
                    if(distances[j] < hit_dist && instances[j][A] == target_type){
                        hit = j;
                        hit_dist = distances[j];
                    }

                    if(distances[j] < mis_dist && instances[j][A] != target_type){
                        mis = j;
                        mis_dist = distances[j];
                    }
                    
                }
                ////////////////////////////////////////////// HIT INSTANCE => instances[hit],  MIS INSTANCE => instances[mis],  TARGET => instances[i]
                
                
                for(int j=0; j<A; j++){                   
                    
                    long double max = instances[0][j];
                    long double min = instances[0][j];
                
                    // max min
                    for(int k=0; k<myInstNum; k++){
                        if(instances[k][j] > max){
                            max = instances[k][j];
                        }    

                        if(instances[k][j] < min){
                            min = instances[k][j];
                        }
                    }// max min 

                    

                  
                    //HIT INSTANCE => instances[hit],  
                    //MIS INSTANCE => instances[mis],  TARGET => instances[i]
                    long double diff1 = fabs(instances[i][j] - instances[hit][j]) / (max-min);
                    long double diff2 = fabs(instances[i][j] - instances[mis][j]) / (max-min);

                    weights[j] = weights[j] - diff1/(long double)M + diff2/(long double)M;

                }

            
            }/// iterations are over and now we need to take T features that has the biggest values

            // first we add the values to the vector<weights[i],i> then sort in terms of weights then take last!!! T amount of that weight index.

            vector<pair<long double,int>> vec;
            for(int j=0; j<A; j++){
                vec.push_back(make_pair(weights[j],j));
            }
            sort(vec.begin(), vec.end());
            
            for(int t=0; t<T; t++){
                selected.push_back(vec[A-t-1].second);
            }
            sort(selected.begin(), selected.end());

            printf("Slave P%d :",rank);
            for(int t=0; t<T; t++){
                printf(" %d",selected[t]);
            }
            printf("\n");
            for(int k=0; k<T; k++){
                selected2[k] = selected[k];
            }


        }

        MPI_Gather(selected2,T,MPI_INT,finalize,T,MPI_INT,0,MPI_COMM_WORLD);


        if(rank == 0){
            set <int>finalize2;
            for(int k=T; k<T*P; k++){
                finalize2.insert(finalize[k]);
            }
            cout<<"Master P0 :";
            for(auto it = finalize2.begin(); it!=finalize2.end(); ++it){
                cout<<" "<<*it;
            }
        }
    // ****************************************** //

    MPI_Barrier(MPI_COMM_WORLD); // synchronizing processes
    MPI_Finalize();

    return 0;
}
