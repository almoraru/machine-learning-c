#include <stdio.h>
#include <stdlib.h>
#include <time.h>



#define train_count sizeof(train)/sizeof(train[0])

// GPT is at trillion of data
// 1 000 000 000 000 => GPT4
// 1 => us

float train[][2] = {
{0, 0},
{1, 2},
{2, 4},
{3, 6},
{4, 8},
};

float rand_float(void)
{
    return((float)rand() / (float)RAND_MAX);
}


// cost = how model behaves. we need to get it close to 0


// We need matrices + derivates
//

float cost(float w)
{
    float result = 0.0f;
    for (size_t i =0; i < train_count;  ++i)
    {
        float x = train[i][0];
        float y = x * w;
        float distance = y - train[i][1];
        // to calculate error UNGA BUNGA style
        result += distance * distance;
        //printf("actual %f | expected %f\n", y, train[i][1]);

    }
    // get average
    result /= train_count;
    return (result);
}

int main()
{
    //y = x * W ?
    //
    //srand(time(0));
    srand(69);
    float w = rand_float() * 10.f;
    // Magical numbers
    float eps = 1e-3; // 1000
    float rate = 1e-3; // 1000
    //printf("error: %f\n", cost(w));
    //printf("error: %f\n", cost(w - eps));
    //printf("error: %f\n", cost(w - eps * 2));
    //printf("Hi\n");
    //printf("%f\n", w);

    // fake derivative
    for (size_t i = 0; i < 500; i++)
    {
        float dcost = (cost (w + eps)  - cost (w)) / eps;
        w -= rate * dcost;
        printf("%f\n", cost(w));
    }

    puts("---");
    printf("%f\n", w);
    return (0);
}
