int plan1d_c2c(int N, int istride, int idist, int ostride, int odist, int batch);
int plan2d_c2c(int N0, int N1, int istride, int idist, int ostride, int odist, int batch);
void execute_c2c(int plan, THCudaTensor *input, THCudaTensor *output, int sign);
int plan2d_r2c(int N0, int N1, int istride, int idist, int ostride, int odist, int batch);
void execute_r2c(int plan, THCudaTensor *input, THCudaTensor *output);
void destroy(int plan);
