#include "micrograd_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void run_labelled(void (*func)(void), char *label)
{
    printf("--- START: %s ---\n", label);
    func();
    printf("--- COMPLETE %s ---\n", label);
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

int main(void)
{
    srand(time(NULL));
    run_labelled(test_value_backpropogation, "value_backpropogation");
    run_labelled(test_neuron_layer, "neuron_layer");
    run_labelled(test_mlp, "mlp_output");
    return 0;
}
