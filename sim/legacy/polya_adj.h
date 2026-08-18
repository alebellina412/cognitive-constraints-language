double time_sec(clock_t START, clock_t STOP)
{
    return (double) (STOP-START) / CLOCKS_PER_SEC;
}
double lrand48norm()
{
    return (double) lrand48() / (1. + RAND_MAX);
}
void make_dir(char *dirname)
{
    mkdir(dirname, 0777);
    errno = 0;
}

double sum(double vec[], int n)
{
    double result=0;
    int i=0;
    for(i=0; i<n; i++){
        result+=vec[i];
    }
    return result;
}

double mean(double vec[], int n)
{
    double result=0;
    int i=0;
    for(i=0; i<n; i++) {
        result+=vec[i];
    }
    return result/n;
}

double variance(double vec[], int n)
{
    double result=0;
    int i=0;
    for(i=0; i<n; i++) {
        result+=vec[i]*vec[i];
    }
    result=result/n-mean(vec,n)*mean(vec,n);
    return result;
}

double mod(double vec[], int n)
{
    double result=0;
    int i=0;
    for(i=0; i<n; i++){
        result+=vec[i]*vec[i];
    }
    return sqrt(result);
}

int sample(double *prob, double r)
{
    // int i=0;
    // double /*ran,*/ cumulative1=0, cumulative2=0;
    // for(i=0;i<n;i++) {
    //     cumulative1=cumulative2;
    //     cumulative2+=prob[i];
    //     if(ran>=cumulative1 && ran<=cumulative2){
    //         break;
    //     }
    // }
    int tmp = 0;
    double cum = prob[tmp];
    while (r > cum)
        cum += prob[++tmp];
    return tmp;
}


double scalar(double v[], double w[], int n)
{
    double result=0;
    for(int i=0; i<n; i++){
        result+=v[i]*w[i];
    }
    return result;
}

double logbase(double a, double base)
{
   return log(a) / log(base);
}

void fprintmatrix(double **matr, int n, int m, FILE **dati)
{
    int i,j;
    for(i=0;i<n;i++) {
        for(j=0;j<m;j++) {
            fprintf(*dati, "%.2lf     ", matr[i][j]);
        }
        fprintf(*dati, "\n");
    }
}

double** transpose(double** M, double** MT, int n, int m)
{
    int i,j;
    for(i=0;i<n;i++) {
        for(j=0;j<m;j++){
            MT[j][i]=M[i][j];
        }
    }
    return MT;
}

double reinforcement2(int rinforzo, double a, int i)
{
    if (rinforzo == 1)
    {
        return log(i + 1);
    }
    if (rinforzo == 2)
    {
        return pow(log(i + 1), a);
    }
    if (rinforzo == 3)
    {
        return pow(i + 1, a);
    }
    if (rinforzo == 4)
    {
        return exp(sqrt(i + 1));
    }
    if (rinforzo == 5)
    {
        return exp(i);
    }
}


int compare(char a[],char b[])  
{  
    int flag=0,i=0;  // integer variables declaration  
    while(a[i]!='\0' &&b[i]!='\0')  // while loop  
    {  
       if(a[i]!=b[i])  
       {  
           flag=1;  
           break;  
       }  
       i++;  
    }  
    if(flag==0)  
    return 1;  
    else  
    return 0;  
}  

double reinforcement(char* rinforzo, double potenza, int t, double coeff)
{

    if (compare(rinforzo,"const"))
    {
        return coeff*potenza;
    }

    if (compare(rinforzo,"log"))
    {
        // printf("%d: è logarimico\n", compare(rinforzo,"log"));
        // printf("%lf\n", pow(log(t+1), potenza));
        return coeff*pow(log(t + 1),potenza);
    }
    if (compare(rinforzo,"pow"))
    {
        // printf("è a potenza\n");
        // printf("%lf\n", pow(t+1, potenza));
        return coeff*pow(t+1, potenza);
    }
    if (compare(rinforzo,"exp"))
    {
        // printf("è esponenziale\n");
        // printf("%d, %lf\n", t, pow(exp(t), potenza));
        return coeff*exp(potenza*t);
    }
}


inline static double sqr(double x) {
    return x*x;
}

int linreg(int n, const double x[], const double y[], double* m, double* b, double* r){
    double   sumx = 0.0;                      /* sum of x     */
    double   sumx2 = 0.0;                     /* sum of x**2  */
    double   sumxy = 0.0;                     /* sum of x * y */
    double   sumy = 0.0;                      /* sum of y     */
    double   sumy2 = 0.0;                     /* sum of y**2  */

    for (int i=0;i<n;i++){ 
        sumx  += x[i];       
        sumx2 += sqr(x[i]);  
        sumxy += x[i] * y[i];
        sumy  += y[i];      
        sumy2 += sqr(y[i]); 
    } 

    double denom = (n * sumx2 - sqr(sumx));
    if (denom == 0) {
        // singular matrix. can't solve the problem.
        *m = 0;
        *b = 0;
        if (r) *r = 0;
            return 1;
    }

    *m = (n * sumxy  -  sumx * sumy) / denom;
    *b = (sumy * sumx2  -  sumx * sumxy) / denom;
    if (r!=NULL) {
        *r = (sumxy - sumx * sumy / n) /    /* compute correlation coeff */
              sqrt((sumx2 - sqr(sumx)/n) *
              (sumy2 - sqr(sumy)/n));
    }

    return 0; 
}

