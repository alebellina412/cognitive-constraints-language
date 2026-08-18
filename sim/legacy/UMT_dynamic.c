#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <inttypes.h>
#include <string.h>
#include "polya_adj.h"

#define NARGC 7

#define N_max pow(10, 7)

int main(int argc, char *argv[])
{
    char fn1[100], fn2[100];
    FILE *dati1, *dati2;

    char *rinforzo_rho, *rinforzo_nu;

    int D0, N0, i, j, *times, estr, insize, D, T, step_print, step_fprint, seed, t_start;

    double ran, *prob, *freq, total, nu, rho, N_objects, N_total, N_objects_temp, N_total_temp, p, alfa, lazy_prob;

    if (argc != NARGC)
    {
        printf("Inserire <N0> <T>\n");
        exit(EXIT_FAILURE);
    }

    N0 = atoi(argv[1]);
    T = atoi(argv[2]);
    rho = atof(argv[3]);
    nu = atof(argv[4]);
    p = atof(argv[5]);
    D0 = atoi(argv[6]);
    alfa = 1. / D0;
    // potenza_rho = atof(argv[5]);
    // potenza_nu = atof(argv[6]);
    // const_rho = atof(argv[7]);
    // const_nu = atof(argv[8]);

    // rho = 1.;
    // nu = 1.;

    seed = 1;

    freq = calloc(N_max, sizeof(*freq));
    times = calloc(N_max, sizeof(*times));
    prob = calloc(N_max, sizeof(*prob));

    N_total = N0;
    N_objects = 1;

    prob[0] = 1.;

    sprintf(fn1, "data/UMT_dynamic_rho=%.1lf_nu=%.1lf_p=%.2lf_N0=%d_D0=%d.dat", rho, nu, p, N0, D0);
    sprintf(fn2, "data/n_UMT_dynamic_rho=%.1lf_nu=%.1lf_p=%.2lf_N0=%d_D0=%d.dat", rho, nu, p, N0, D0);
    dati1 = fopen(fn1, "w");
    dati2 = fopen(fn2, "w");

    // srand48(time(0));
    srand48(seed);

    D = 0;

    // step=round((double)T/100);
    step_print = 1000;
    step_fprint = 1;

    double prob_sat = 0;

    double total_old = 0, total_new = N0;

    for (int t = 0; t < T; t++)
    {

        ran = (lrand48() / (RAND_MAX + 1.0));
        // lazy_prob = p * (1 - 1 / pow(1 + alfa, D));
        lazy_prob = p * (pow(alfa * D, 2) / (pow(alfa * D, 2) + 1 ));

        if(ran < (1 - lazy_prob)) 
        {
            // printf("t = %d pesco dall'urna || t_start = %d, ran = %lf < (1-p) = %lf\n", t, t_start, ran, 1-p);

            // printf("estraggo \n");

            ran = (lrand48() / (RAND_MAX + 1.0));
            estr = sample(prob, ran);

            // printf("estratto \n");

            if (t % step_print == 0)
            {
                printf("t = %d\tD = %d\testr=%d\n", t, D, estr);
            }
            if (t < step_fprint || t % step_fprint == 0)
            {
                fprintf(dati1, "%d\t%d\t%d\n", t, D, estr);
            }

            if (estr == N_objects - 1)
            {

                // printf("estratto elemento nuovo: probabilità = %lf \n", prob[estr]);

                times[estr] = t;
                D++;

                prob[estr + 1] = prob[estr];

                N_objects++;
                N_total_temp = N_total;

                // add_old = reinforcement(rinforzo_rho, potenza_rho, t, const_rho);

                freq[estr] = 1;

                // add_new = reinforcement(rinforzo_nu, potenza_nu, D, const_nu);

                N_total += rho;
                N_total += nu;

                prob[estr] = rho / N_total_temp;
                prob[estr + 1] += (nu) / N_total_temp;

                /*** COMPUTE ANALITYCAL PROBABILITIES ***/
                // total_old += add_old;
                // total_new += add_new;
                /*** CHECK PROBABILITIES ***/
                // printf("t = %d \t D = %d, total_old = %lf, add_old = %lf, total_new = %lf, add_new = %lf, N_total = %lf\n", t, D, total_old, total_new, N_total);
                /*** END CHECK ***/
            }
            else
            {
                N_total_temp = N_total;

                // add_old = reinforcement(rinforzo_rho, potenza_rho, t, const_rho);

                N_total += rho;

                freq[estr] += 1;

                prob[estr] += rho / N_total_temp;

                /*** COMPUTE ANALITYCAL PROBABILITIES ***/
                // total_old += add_old;
                // printf("t = %d \t D = %d, total_old = %lf, add_old = %lf, total_new = %lf, N_total = %lf\n", t, D, total_old, add_old, total_new, N_total);
            }

            for (i = 0; i < N_objects; i++)
            {
                prob[i] *= (N_total_temp / N_total);
            }

        }
        else
        {

            // printf("t = %d pesco dallo stream \n", t);

            for(i = 0; i < N_objects; i++)
            {
                freq[i] /= t;
            }

            ran = (lrand48() / (RAND_MAX + 1.0));
            estr = sample(freq, ran);

            for(i = 0; i < N_objects; i++)
            {
                freq[i] *= t;
            }

            if (t % step_print == 0)
            {
                printf("t = %d\tD = %d\testr=%d\n", t, D, estr);
            }
            if (t < step_fprint || t % step_fprint == 0)
            {
                fprintf(dati1, "%d\t%d\t%d\n", t, D, estr);
            }

            N_total_temp = N_total;

            // add_old = reinforcement(rinforzo_rho, potenza_rho, t, const_rho);

            N_total += rho;

            freq[estr] += 1;

            prob[estr] += rho / N_total_temp;

            for (i = 0; i < N_objects; i++)
            {
                prob[i] *= (N_total_temp / N_total);
            }

        }
    }

    /*** CHECK PROBABILITIES ***/

    // double prob_old = 0., prob_new = 0.;

    // for (i = 0; i < N_objects - 1; i++)
    // {
    //     prob_old += prob[i];
    // }

    // prob_new = prob[(int)(N_objects - 1)];
    // printf("t = %d \t analytical probabilities: prob_old = %lf \t prob_new = %lf \t norm = %lf\n", T, total_old / N_total, total_new / N_total, total_old / N_total + total_new / N_total);
    // printf("t = %d \t measured probabilities: prob_old = %lf \t prob_new = %lf \t norm = %lf\n", T, prob_old, prob_new, prob_old + prob_new);

    /*** END CHECK ***/

    for (i = 0; i < N_objects; i++)
    {
        fprintf(dati2, "%d\t%lf\t%lf\t%lf\n", times[i], freq[i], prob[i] * N_total, prob[i]);
    }

    fclose(dati1);
    fclose(dati2);
    free(prob);
    free(freq);
    free(times);
}
