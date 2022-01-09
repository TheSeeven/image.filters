#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "global_variables_functions.h"
#include "cuda_algorithms.h"


/* main function separates all the phases of the program in 4 big categories.

Init: Prepares all the data needed in order for the GPU to have something to work with.
    Alocates memory and copies the required data.

start_processing: In this phase the gpu will work hard to create some beautifull results :)

save_processing: We save the progress, so we can post to instagram later :)

End: We need to clear all the used memory and prepare the program to be closed in a 
    safely manner without memory leakages */

int main() {
    Init();
    start_processing();
    save_processing();
    End();
    return 0;
}