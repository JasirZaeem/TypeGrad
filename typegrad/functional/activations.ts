import { Value } from "@/typegrad";

export const identity = (a: Value) => {
  const out = new Value(a.value, [a], "identity");
  out._backward = () => {
    a.grad += out.grad;
  };
  return out;
};

export const sigmoid = (a: Value) => {
  const out = new Value(1 / (1 + Math.exp(-a.value)), [a], "sigmoid");
  out._backward = () => {
    a.grad += out.grad * out.value * (1 - out.value);
  };
  return out;
};

export const relu = (a: Value) => {
  const out = new Value(Math.max(0, a.value), [a], "relu");
  out._backward = () => {
    a.grad += out.grad * (out.value > 0 ? 1 : 0);
  };
  return out;
};

export const tanh = (a: Value) => {
  const out = new Value(Math.tanh(a.value), [a], "tanh");
  out._backward = () => {
    a.grad += out.grad * (1 - out.value * out.value);
  };
  return out;
};

export enum Activations {
  Identity,
  Sigmoid,
  Relu,
  Tanh,
}
