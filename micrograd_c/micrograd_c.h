#ifndef MICROGRAD_C_H
#define MICROGRAD_C_H
#include <stdint.h>
// -- VALUE headers
typedef struct Value
{
    double data;
    double grad;
    struct Value *_prev[2];
    char *_op;
    char *_label;
    double _aux;
    void (*_backward)(struct Value *self);
    int pinned; // if it should be freed or not
    int visited;
} Value;

Value *new_value(double data);
Value **new_values(double *data, uint64_t len);
void free_value(Value *v);

void set_data(Value *v, double data);
void set_grad(Value *v, double grad);
void set_prev(Value *v, Value *prev[2]);
void set_op(Value *v, char *op);
void set_backward(Value *v, void (*backward)(Value *self));
void set_label(Value *v, char *label);

Value *add_values(Value *a, Value *b);
Value *mul_values(Value *a, Value *b);

Value *tanh_value(Value *a);
Value *relu_value(Value *a);
Value *exp_value(Value *a);
Value *pow_value(Value *a, double exp);

typedef Value *(*activation_fn)(Value *);

void print_value(Value *v);

void backward(Value *out);

// NEURON headers
typedef struct
{
    Value **weights;
    Value *bias;
    uint64_t nin;
} Neuron;

Neuron *new_neuron(Value **weights, uint64_t nin, double bias);
Neuron *new_rand_neuron(uint64_t nin);
void free_neuron(Neuron *n);

void set_weights(Neuron *n, Value **weights);
void set_bias(Neuron *n, Value *bias);
void set_nin(Neuron *n, uint64_t nin);

Value *compute_activation(Neuron *n, Value **inputs, activation_fn fn);

// LAYER headers
typedef struct
{
    Neuron **neurons;
    uint64_t nout;
} Layer;

Layer *new_layer(Neuron **neurons, uint64_t nout);
Layer *new_rand_layer(uint64_t nin, uint64_t nout);
Value **compute_activations(Layer *l, Value **inputs, activation_fn fn);

#endif

// MLP headers

typedef struct
{
    Layer **layers;
    uint64_t nlayers;
    Value **out;
} MLP;

MLP *new_mlp(Layer **layers, uint64_t nlayers);
MLP *new_rand_mlp(uint64_t nin, uint64_t nouts[], uint64_t nlayers);
Value **compute_outputs_mlp(MLP *m, Value **inputs, activation_fn fn);

Value *mse_loss(Value **ypreds, Value **ys, uint64_t len);
Value **zero_grad(Value **params, uint64_t nparams);
Value **collect_parameters(MLP *m, uint64_t *nparams);
