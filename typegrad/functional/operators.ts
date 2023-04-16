import { Value } from "@/typegrad";

// Math operations

export const add = (a: Value, b: Value) => {
  const out = new Value(a.value + b.value, [a, b], `+`);
  out._backward = () => {
    a.grad += out.grad;
    b.grad += out.grad;
  };
  return out;
};

export const sub = (a: Value, b: Value) => {
  const out = new Value(a.value - b.value, [a, b], `-`);
  out._backward = () => {
    a.grad += out.grad;
    b.grad += -out.grad;
  };
  return out;
};

export const neg = (a: Value) => {
  const out = new Value(-a.value, [a], `-ve`);
  out._backward = () => {
    a.grad += -out.grad;
  };
  return out;
};

export const mul = (a: Value, b: Value) => {
  const out = new Value(a.value * b.value, [a, b], `*`);
  out._backward = () => {
    a.grad += out.grad * b.value;
    b.grad += out.grad * a.value;
  };
  return out;
};

export const div = (a: Value, b: Value) => {
  const out = new Value(a.value / b.value, [a, b], `/`);
  out._backward = () => {
    a.grad += out.grad / b.value;
    b.grad += (-a.value / (b.value * b.value)) * out.grad;
  };
  return out;
};

export const pow = (a: Value, b: number) => {
  const out = new Value(Math.pow(a.value, b), [a], `^`);
  out._backward = () => {
    a.grad += out.grad * b * Math.pow(a.value, b - 1);
  };
  return out;
};

export const exp = (a: Value) => {
  const out = new Value(Math.exp(a.value), [a], `e^x`);
  out._backward = () => {
    a.grad += out.grad * out.value;
  };
  return out;
};
