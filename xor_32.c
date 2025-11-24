/* ************************************************************************** */
/*                                                                            */
/*                        ______                                              */
/*                     .-"      "-.                                           */
/*                    /            \                                          */
/*        _          |              |          _                              */
/*       ( \         |,  .-.  .-.  ,|         / )                             */
/*        > "=._     | )(__/  \__)( |     _.=" <                              */
/*       (_/"=._"=._ |/     /\     \| _.="_.="\_)                             */
/*              "=._ (_     ^^     _)"_.="                                    */
/*                  "=\__|IIIIII|__/="                                        */
/*                 _.="| \IIIIII/ |"=._                                       */
/*       _     _.="_.="\          /"=._"=._     _                             */
/*      ( \_.="_.="     `--------`     "=._"=._/ )                            */
/*       > _.="                            "=._ <                             */
/*      (_/                                    \_)                            */
/*                                                                            */
/*      Filename: main.c                                                      */
/*      By: espadara <espadara@pirate.capn.gg>                                */
/*      Created: 2025/11/24 09:00:39 by espadara                              */
/*      Updated: 2025/11/24 09:05:52 by espadara                              */
/*                                                                            */
/* ************************************************************************** */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#define HIDDEN_NEURONS 64

// --- MATRIX ENGINE ---
typedef struct { size_t rows, cols; float *es; } Mat;
#define MAT_AT(m, i, j) (m).es[(i)*(m).cols + (j)]

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m; m.rows = rows; m.cols = cols;
    m.es = calloc(rows * cols, sizeof(*m.es));
    return m;
}

// Optimized Zeroing
void mat_zero(Mat m) {
    memset(m.es, 0, sizeof(*m.es) * m.rows * m.cols);
}

void mat_rand(Mat m) {
    for (size_t i = 0; i < m.rows * m.cols; ++i)
        m.es[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

void mat_dot(Mat dst, Mat a, Mat b) {
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < a.cols; ++k)
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
        }
    }
}

void mat_sum(Mat dst, Mat a) {
    for (size_t i = 0; i < dst.rows * dst.cols; ++i) dst.es[i] += a.es[i];
}

void mat_sig(Mat m) {
    for (size_t i = 0; i < m.rows * m.cols; ++i)
        m.es[i] = 1.f / (1.f + expf(-m.es[i]));
}

// --- NETWORK ---

typedef struct {
    Mat w1, b1, w2, b2;       // Parameters
    Mat dw1, db1, dw2, db2;   // Gradients
    Mat a0, a1, a2;           // Cache
} Xor;

Xor xor_alloc(void) {
    Xor m;
    // Input (2) -> Hidden (32) -> Output (1)
    m.w1 = mat_alloc(2, HIDDEN_NEURONS); m.b1 = mat_alloc(1, HIDDEN_NEURONS);
    m.w2 = mat_alloc(HIDDEN_NEURONS, 1); m.b2 = mat_alloc(1, 1);

    m.dw1 = mat_alloc(2, HIDDEN_NEURONS); m.db1 = mat_alloc(1, HIDDEN_NEURONS);
    m.dw2 = mat_alloc(HIDDEN_NEURONS, 1); m.db2 = mat_alloc(1, 1);

    m.a0 = mat_alloc(1, 2);
    m.a1 = mat_alloc(1, HIDDEN_NEURONS);
    m.a2 = mat_alloc(1, 1);

    mat_rand(m.w1); mat_rand(m.b1);
    mat_rand(m.w2); mat_rand(m.b2);
    return m;
}

void forward(Xor m, float x1, float x2) {
    MAT_AT(m.a0, 0, 0) = x1; MAT_AT(m.a0, 0, 1) = x2;
    mat_dot(m.a1, m.a0, m.w1); mat_sum(m.a1, m.b1); mat_sig(m.a1);
    mat_dot(m.a2, m.a1, m.w2); mat_sum(m.a2, m.b2); mat_sig(m.a2);
}

void backprop(Xor m, float target) {
    float y = MAT_AT(m.a2, 0, 0);
    float output_grad = 2.f * (y - target) * (y * (1.f - y));

    // Update Output Layer Gradients
    for(size_t i = 0; i < HIDDEN_NEURONS; ++i) {
        MAT_AT(m.dw2, i, 0) += MAT_AT(m.a1, 0, i) * output_grad;
    }
    MAT_AT(m.db2, 0, 0) += output_grad;

    // Update Hidden Layer Gradients
    for(size_t i = 0; i < HIDDEN_NEURONS; ++i) {
        float val = MAT_AT(m.a1, 0, i);
        float hidden_grad = (output_grad * MAT_AT(m.w2, i, 0)) * (val * (1.f - val));

        for(size_t j = 0; j < 2; ++j) {
            MAT_AT(m.dw1, j, i) += MAT_AT(m.a0, 0, j) * hidden_grad;
        }
        MAT_AT(m.db1, 0, i) += hidden_grad;
    }
}

// Clean and Fast
void clear_grads(Xor m) {
    mat_zero(m.dw1);
    mat_zero(m.db1);
    mat_zero(m.dw2);
    mat_zero(m.db2);
}

void update(Xor m, float rate) {
    for(size_t i=0; i<2*HIDDEN_NEURONS; ++i) m.w1.es[i] -= rate * m.dw1.es[i];
    for(size_t i=0; i<HIDDEN_NEURONS; ++i) m.b1.es[i] -= rate * m.db1.es[i];
    for(size_t i=0; i<HIDDEN_NEURONS; ++i) m.w2.es[i] -= rate * m.dw2.es[i];
    for(size_t i=0; i<1; ++i) m.b2.es[i] -= rate * m.db2.es[i];
}

int main(void) {
    srand(69); // Nice
    Xor m = xor_alloc();
    float td[][3] = {{0,0,0}, {0,1,1}, {1,0,1}, {1,1,0}};

    printf("Training 64-Neuron Behemoth...\n");
    for (size_t i = 0; i < 100042; ++i) {
        // Reset gradients at the start of the batch
        clear_grads(m);

        for (size_t j = 0; j < 4; ++j) {
            forward(m, td[j][0], td[j][1]);
            backprop(m, td[j][2]);
        }
        update(m, 1.0f);
        if (i % 2000 == 0) printf("Cost: %f\n", MAT_AT(m.a2,0,0) - td[3][2]);
    }

    printf("\n--- Results (The 64 Neurons have spoken) ---\n");
    for (size_t j = 0; j < 4; ++j) {
        forward(m, td[j][0], td[j][1]);
        printf("%.0f ^ %.0f = %f\n", td[j][0], td[j][1], MAT_AT(m.a2, 0, 0));
    }
    return 0;
}
