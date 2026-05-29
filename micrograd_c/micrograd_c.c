// my attempt at a simple autograd implementation
// in C to follow along with the micrograd series

// v1: my attempt based on getting about an hour into the video

// v2: fixing the bug discussed where we overwrite gradients of children
// when both children (fields in prev) are pointers to the same Value

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// struct represents a value in the graph
typedef struct Value
{
    double data;
    double grad;
    struct Value *_prev[2];
    char *_op;
    char *_label;
    double _aux;
    void (*_backward)(struct Value *self);
} Value;

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

Value *tanh_values(Value *a)
{
    Value *prev[2] = {a, NULL};
    Value *v = new_value(tanh(a->data));
    set_op(v, "tanh");
    set_prev(v, prev);
    set_backward(v, tanh_backward);
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

int main(void)
{
}
