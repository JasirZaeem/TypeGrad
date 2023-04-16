import { functional as fn } from "@/typegrad";

export class Value {
  value: number;
  prev: Set<Value>;
  op: string;
  label: string;
  grad: number;
  _backward: () => void;

  constructor(value: number, _prev: Value[] = [], _op = "", label = "") {
    this.value = value * 1.0;
    this.grad = 0.0;
    this.prev = new Set(_prev);
    this.op = _op;
    this.label = label;
    this._backward = () => undefined;
  }

  static from(value: number, label = "") {
    return new Value(value, [], "", label);
  }

  backward() {
    // topological sort prev
    const sorted: Value[] = [];
    const visited = new Set();

    const visit = (node: Value) => {
      if (visited.has(node)) {
        return;
      }
      visited.add(node);
      for (const prev of node.prev) {
        visit(prev);
      }
      sorted.push(node);
    };

    visit(this);

    // backward pass
    this.grad = 1.0;

    for (let i = sorted.length - 1; i >= 0; i--) {
      sorted[i]._backward();
    }
  }

  // Math operators
  add(other: Value) {
    return fn.add(this, other);
  }

  neg() {
    return fn.neg(this);
  }

  sub(other: Value) {
    return fn.sub(this, other);
  }

  mul(other: Value) {
    return fn.mul(this, other);
  }

  div(other: Value) {
    return fn.div(this, other);
  }

  pow(other: number) {
    return fn.pow(this, other);
  }

  exp() {
    return fn.exp(this);
  }

  // Activation functions

  sigmoid() {
    return fn.sigmoid(this);
  }

  relu() {
    return fn.relu(this);
  }

  tanh() {
    return fn.tanh(this);
  }

  // Util

  toString() {
    return fn.toString(this);
  }

  printComputationGraph() {
    return fn.printComputationGraph(this);
  }
}

export const v = Value.from;
