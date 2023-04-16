import { Value } from "@/typegrad";

export const printComputationGraph = (value: Value, indent = 0) => {
  console.log(" ".repeat(indent) + value.toString());
  for (const val of value.prev) {
    printComputationGraph(val, indent + 2);
  }
};

/**
 * Convert an array of numbers to an array of Value objects.
 */
export const fromArray = (arr: number[]): Value[] => {
  const out: Value[] = new Array(arr.length);
  for (let i = 0; i < arr.length; ++i) {
    out[i] = new Value(arr[i]);
  }
  return out;
};

export const fromMatrix = (arr: number[][]): Value[][] => {
  const out: Value[][] = new Array(arr.length);
  for (let i = 0; i < arr.length; ++i) {
    out[i] = fromArray(arr[i]);
  }
  return out;
};

type ValueContainerImmediate =
  | Value[]
  | Iterable<Value>
  | Record<string, Value>
  | Value;
export type ValueContainer =
  | ValueContainerImmediate
  | ValueContainerImmediate[]
  | Iterable<ValueContainerImmediate>
  | Record<string, ValueContainerImmediate>;

/**
 * Returns a generator that yields all values in a arbitrarily nested
 * structure of Value objects.
 * @param values
 */
export function* getValues(
  values: ValueContainer
): Generator<Value, void, undefined> {
  if (values instanceof Value) {
    yield values;
  } else if (Array.isArray(values)) {
    for (const value of values) {
      yield* getValues(value);
    }
  } else if (isIterable<Value>(values)) {
    for (const value of values) {
      yield* getValues(value);
    }
  } else {
    for (const key in values) {
      yield* getValues((values as Record<string, ValueContainer>)[key]);
    }
  }
}

export function* runBatch<T, U>(
  model: { forward: (item: T) => U },
  batch: T[]
): Generator<U, void, undefined> {
  for (const item of batch) {
    yield model.forward(item);
  }
}

/**
 * Creates an array from an iterable, or returns the iterable if it is already an array.
 * @param iter
 */
export const toArray = <T>(iter: Iterable<T>): T[] =>
  Array.isArray(iter) ? iter : (Array.from(iter) as T[]);

export const isIterable = <T>(iter: unknown): iter is IterableIterator<T> => {
  return Symbol.iterator in Object(iter);
};

export const toString = (v: Value) => {
  return `Value(${Number.isInteger(v.value) ? v.value.toFixed(1) : v.value}${
    v.label ? ` (${v.label})` : ""
  }${v.op ? ` ${v.op}` : ""}${
    v.grad
      ? ` grad: ${Number.isInteger(v.grad) ? v.grad.toFixed(1) : v.grad}`
      : ""
  })`;
};
