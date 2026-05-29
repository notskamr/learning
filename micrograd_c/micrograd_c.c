// my attempt at a simple autograd implementation
// in C to follow along with the micrograd series

// v1: my attempt based on getting about an hour into the video

// v2: fixing the bug discussed where we overwrite gradients of children
// when both children (fields in prev) are pointers to the same Value

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "micrograd_c.h"

// -- VALUE definitions

// no-op default backward operation
void noop_backward(Value *self)
{
    return;
}

// create a new value
Value *new_value(double data)
{
    Value *v = (Value *)malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->_label = NULL;
    v->_backward = noop_backward;
    v->_op = NULL;
    v->_prev[0] = NULL;
    v->_prev[1] = NULL;
    return v;
}

Value **new_values(double *data, uint64_t len)
{
    Value **vals = (Value **)malloc(sizeof(Value *) * len);
    for (uint64_t i = 0; i < len; i++)
    {
        vals[i] = new_value(data[i]);
    }
    return vals;
}

void free_value(Value *v)
{
    if (v == NULL)
        return;
    free_value(v->_prev[0]);
    free_value(v->_prev[1]);
    free(v);
}

void set_data(Value *v, double data)
{
    if (v == NULL)
        return;
    v->data = data;
}

void set_grad(Value *v, double grad)
{
    if (v == NULL)
        return;
    v->grad = grad;
}

void set_prev(Value *v, Value *prev[2])
{
    if (v == NULL)
        return;
    v->_prev[0] = prev[0];
    v->_prev[1] = prev[1];
}

void set_op(Value *v, char *op)
{
    if (v == NULL)
        return;
    v->_op = op;
}

void set_backward(Value *v, void (*backward)(Value *self))
{
    if (v == NULL)
        return;
    v->_backward = backward;
}

void set_label(Value *v, char *label)
{
    if (v == NULL)
        return;
    v->_label = label;
}

// backward propogation - chain rules for each operation
void add_backward(Value *self)
{
    if (self->_prev[0])
        self->_prev[0]->grad += self->grad; // v2 fix: += instead of +
    if (self->_prev[1])
        self->_prev[1]->grad += self->grad;
}

void mul_backward(Value *self)
{
    if (self->_prev[0] && self->_prev[1])
    {
        self->_prev[0]->grad += self->_prev[1]->data * self->grad;
        self->_prev[1]->grad += self->_prev[0]->data * self->grad;
    }
}

void tanh_backward(Value *self)
{
    // d/dx tanh(x) = 1 - (tanh(x))^2 -> tanh(x)^2 is just self->data
    if (self->_prev[0])
        self->_prev[0]->grad += (1.0 - self->data * self->data) * self->grad;
}

void relu_backward(Value *self)
{
    if (self->_prev[0])
        self->_prev[0]->grad += (self->_prev[0]->data > 0.0 ? 1.0 : 0.0) * self->grad;
}

void exp_backward(Value *self)
{
    // d/dx e^x = e^x -> self->data
    if (self->_prev[0])
        self->_prev[0]->grad += self->data * self->grad;
}

void pow_backward(Value *self)
{
    // d/dx (x^n) = n * x^(n-1)
    if (self->_prev[0])
        self->_prev[0]->grad += self->_aux * pow(self->_prev[0]->data, self->_aux - 1) * self->grad;
}

// operations

Value *add_values(Value *a, Value *b)
{
    Value *prev[2] = {a, b};
    Value *v = new_value(a->data + b->data);
    set_op(v, "+");
    set_prev(v, prev);
    set_backward(v, add_backward);
    return v;
}

Value *mul_values(Value *a, Value *b)
{
    Value *prev[2] = {a, b};
    Value *v = new_value(a->data * b->data);
    set_op(v, "*");
    set_prev(v, prev);
    set_backward(v, mul_backward);
    return v;
}

Value *tanh_value(Value *a)
{
    Value *prev[2] = {a, NULL};
    Value *v = new_value(tanh(a->data));
    set_op(v, "tanh");
    set_prev(v, prev);
    set_backward(v, tanh_backward);
    return v;
}

Value *relu_value(Value *a)
{
    Value *prev[2] = {a, NULL};
    Value *v = new_value(fmax(0.0, a->data));
    set_op(v, "relu");
    set_prev(v, prev);
    set_backward(v, relu_backward);
    return v;
}

Value *exp_value(Value *a)
{
    Value *prev[2] = {a, NULL};
    Value *v = new_value(exp(a->data));
    set_op(v, "exp");
    set_prev(v, prev);
    set_backward(v, exp_backward);
    return v;
}

// a ^ b
Value *pow_value(Value *a, double exp)
{
    Value *prev[2] = {a, NULL};
    Value *v = new_value(pow(a->data, exp));
    v->_aux = exp;
    set_op(v, "pow");
    set_prev(v, prev);
    set_backward(v, pow_backward);
    return v;
}

// topological sorter - recursive
void build_topo(Value *v, Value **topo, unsigned int *t_len, Value **visited, unsigned int *v_len)
{
    if (v == NULL)
        return;

    // set of visited nodes - only consider if not visited
    for (int i = 0; i < *v_len; i++)
        if (visited[i] == v)
            return;

    // add unvisited node to arr
    visited[(*v_len)++] = v;

    // recurse through children, build all children (all dependencies)
    // before adding current node to topo chart -> dependencies always
    // placed before, hence we can backpropogate
    for (int i = 0; i < 2; i++)
    {
        build_topo(v->_prev[i], topo, t_len, visited, v_len);
    }

    // add current node to topo at right spot
    topo[(*t_len)++] = v;
}

// go backwards in topological order chaining
void backward(Value *out)
{
    // get an array to store topological order
    Value *topo[2000];

    // visited store
    Value *visited[2000];

    unsigned int t_len = 0;
    unsigned int v_len = 0;

    // seed output gradient - always 1: df/df = 1
    out->grad = 1.0;

    build_topo(out, topo, &t_len, visited, &v_len);

    // iterate backwards through the topological order
    for (unsigned int i = t_len; i-- > 0;)
    {
        topo[i]->_backward(topo[i]);
    }
}

void print_value(Value *v)
{
    printf("Value(data=%f, grad=%f, op=%s, label=%s)\n", v->data, v->grad, v->_op ? v->_op : "None", v->_label ? v->_label : "None");
}

// -- NEURON definitions

double rand_double(double min, double max)
{
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

Neuron *new_neuron(Value **weights, uint64_t nin, double bias)
{
    Neuron *n = (Neuron *)malloc(sizeof(Neuron));
    n->weights = weights;
    n->nin = nin;
    n->bias = new_value(bias);
    return n;
}

Neuron *new_rand_neuron(uint64_t nin)
{
    Value **weights = (Value **)malloc(sizeof(Value *) * nin);
    double bias = rand_double(-1.0, 1.0);
    for (uint64_t i = 0; i < nin; i++)
    {
        weights[i] = new_value(rand_double(-1.0, 1.0));
    }
    return new_neuron(weights, nin, bias);
}

void free_neuron(Neuron *n)
{
    if (n == NULL)
        return;
    for (uint64_t i = 0; i < n->nin; i++)
        free_value(n->weights[i]);
    free(n->weights);
    free_value(n->bias);
    free(n);
}

Value *compute_activation(Neuron *n, Value **inputs, activation_fn fn)
{
    Value *out = n->bias;
    for (uint64_t i = 0; i < n->nin; i++)
    {
        out = add_values(out, mul_values(n->weights[i], inputs[i]));
    }
    return fn(out);
}

// LAYER definitions

Layer *new_layer(Neuron **neurons, uint64_t nout)
{
    Layer *l = (Layer *)malloc(sizeof(Layer));
    l->neurons = neurons;
    l->nout = nout;
    return l;
}

Layer *new_rand_layer(uint64_t nin, uint64_t nout)
{
    // nin is dimensions per neuron, nout is count of neurons
    // each neuron owns its dimensions, the layer just owns count of neurons
    Neuron **neurons = malloc(sizeof(Neuron *) * nout);
    for (uint64_t i = 0; i < nout; i++)
    {
        neurons[i] = new_rand_neuron(nin);
    }
    return new_layer(neurons, nout);
}

Value **compute_activations(Layer *l, Value **inputs, activation_fn fn)
{
    Value **outs = malloc(sizeof(Value *) * l->nout);
    for (uint64_t i = 0; i < l->nout; i++)
    {
        outs[i] = compute_activation(l->neurons[i], inputs, fn);
    }
    return outs;
}

// MLP definitions

MLP *new_mlp(Layer **layers, uint64_t nlayers)
{
    MLP *m = (MLP *)malloc(sizeof(MLP));
    m->layers = layers;
    m->nlayers = nlayers;
    m->out = NULL;
    return m;
}

MLP *new_rand_mlp(uint64_t nin, uint64_t nouts[], uint64_t nlayers)
{
    // nin is dimensions of input, nouts is array of count of neurons in each layer
    // each layer owns its count of neurons, the MLP just owns the layers
    // the mlp has a total of nlayers

    Layer **layers = malloc(sizeof(Layer *) * nlayers);
    uint64_t cur = nin;
    for (uint64_t i = 0; i < nlayers; i++)
    {
        layers[i] = new_rand_layer(cur, nouts[i]);
        cur = nouts[i];
    }
    return new_mlp(layers, nlayers);
}

Value **compute_outputs_mlp(MLP *m, Value **inputs, activation_fn fn)
{
    for (uint64_t i = 0; i < m->nlayers; i++)
    {
        Value **tmp = inputs;
        inputs = compute_activations(m->layers[i], inputs, fn);
        if (i != 0)
            free(tmp);
    }
    m->out = inputs;
    return inputs;
}
