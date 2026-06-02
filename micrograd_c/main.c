#include "micrograd_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

void run_labelled(void (*func)(void), char *label)
{
    printf("--- START: %s ---\n", label);
    func();
    printf("--- COMPLETE %s ---\n", label);
}

void pin_values(Value **vals, uint64_t len)
{
    for (uint64_t i = 0; i < len; i++)
    {
        vals[i]->pinned = 1;
    }
}

Value ***create_inputs_from_array(double **data, uint64_t nrows, uint64_t ncols)
{
    Value ***inputs = (Value ***)malloc(sizeof(Value **) * nrows);
    for (uint64_t i = 0; i < nrows; i++)
    {
        inputs[i] = new_values(data[i], ncols);
        pin_values(inputs[i], ncols);
    }
    return inputs;
}

void test_value_backpropogation(void)
{
    Value *x1 = new_value(2.0);
    Value *x2 = new_value(0.0);
    Value *w1 = new_value(-3.0);
    Value *w2 = new_value(1.0);

    Value *b = new_value(6.88137358);

    Value *n = add_values(add_values(mul_values(x1, w1), mul_values(x2, w2)), b);
    Value *o = tanh_value(n);

    set_label(x1, "x1");
    set_label(x2, "x2");
    set_label(w1, "w1");
    set_label(w2, "w2");
    set_label(b, "b");
    set_label(n, "n");
    set_label(o, "o");

    backward(o);

    print_value(x2);
    print_value(w2);
    print_value(x1);
    print_value(w1);
}

void test_neuron_layer(void)
{
    int nin = 2;
    int nout = 3;
    double x[] = {2.0, 3.0};
    Value **x_vals = new_values(x, nin);
    Layer *l = new_rand_layer(nin, nout);
    Value **outs = compute_activations(l, x_vals, tanh_value);
    for (int i = 0; i < nout; i++)
    {
        print_value(outs[i]);
    }
}

void test_mlp(void)
{
    uint64_t nin = 3;
    uint64_t nouts[] = {4, 4, 1};
    double x[] = {2.0, 3.0, 1.0};
    Value **inputs = new_values(x, nin);
    MLP *m = new_rand_mlp(nin, nouts, 3);
    compute_outputs_mlp(m, inputs, tanh_value);
    for (int i = 0; i < m->layers[m->nlayers - 1]->nout; i++)
    {
        print_value(m->out[i]);
    }
}

void test_pred(void)
{

    uint64_t nin = 3;
    uint64_t nouts[] = {4, 4, 1};
    MLP *m = new_rand_mlp(nin, nouts, 3);
    uint64_t nrows = 4;
    uint64_t ncols = nin;
    double xs_vals[4][3] = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}};

    double *xs_ptrs[4];
    for (int i = 0; i < 4; i++)
        xs_ptrs[i] = xs_vals[i];

    double ys_vals[] = {1.0, -1.0, -1.0, -1.0}; // desired

    Value ***inputs = create_inputs_from_array(xs_ptrs, nrows, ncols);
    Value **ys = new_values(ys_vals, nrows);

    Value **ypreds = malloc(sizeof(Value *) * nrows);
    for (uint64_t i = 0; i < nrows; i++)
    {
        Value **out = compute_outputs_mlp(m, inputs[i], tanh_value);
        ypreds[i] = out[0];
        ys[i]->pinned = 1;
        print_value(ypreds[i]);
        free(out);
    }

    Value *loss = mse_loss(ypreds, ys, nrows);
    printf("loss: %f\n", loss->data);

    uint64_t nparams;
    Value **params = collect_parameters(m, &nparams);
    printf("nparams: %llu\n", nparams);

    double step = 0.01;
    uint64_t passes = 1000;
    for (uint64_t pass = 0; pass < passes; pass++)
    {

        // forward pass
        for (uint64_t i = 0; i < nrows; i++)
        {
            if (ypreds[i])
                ypreds[i]->pinned = 0; // unpin old
            Value **out = compute_outputs_mlp(m, inputs[i], tanh_value);
            ypreds[i] = out[0];
            free(out);
        }
        Value *prev = loss;
        loss = mse_loss(ypreds, ys, nrows);

        // backward pass
        zero_grad(params, nparams);
        backward(loss);

        // gradient descent
        for (uint64_t i = 0; i < nparams; i++)
        {
            params[i]->data += -step * params[i]->grad;
        }

        if (pass % 10 == 0)
            printf("pass: %llu, loss: %f\n", pass, loss->data);

        int diverged = loss->data > prev->data;
        if (diverged)
            printf("WARNING: loss increased from %f to %f\n", prev->data, loss->data);

        free_value(prev);
        if (diverged)
            break;
    }

    printf("final loss: %f\n", loss->data);
    for (uint64_t i = 0; i < nrows; i++)
    {
        print_value(ypreds[i]);
    }
}

int main(void)
{
    srand((unsigned int)(time(NULL) ^ getpid()));
    run_labelled(test_value_backpropogation, "value_backpropogation");
    run_labelled(test_neuron_layer, "neuron_layer");
    run_labelled(test_mlp, "mlp_output");
    run_labelled(test_pred, "pred");
    return 0;
}
