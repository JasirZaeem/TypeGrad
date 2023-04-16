import * as tg from "@/typegrad";

export const sum = (values: Iterable<tg.Value>, init?: tg.Value) => {
  const valueArray = tg.toArray(values);
  let outVal = init?.value ?? 0;
  for (const val of valueArray) {
    outVal += val.value;
  }
  const out = new tg.Value(
    outVal,
    valueArray,
    `sum (${valueArray.length} values)`
  );

  out._backward = () => {
    for (const val of valueArray) {
      val.grad += out.grad;
    }
    if (init) {
      init.grad += out.grad;
    }
  };
  return out;
};

export const mean = (values: Iterable<tg.Value>) => {
  const valueArray = tg.toArray(values);
  const sum = tg.sum(valueArray);
  const out = sum.div(new tg.Value(valueArray.length, [], "mean divisor"));

  out._backward = () => {
    for (const val of valueArray) {
      val.grad += out.grad / valueArray.length;
    }
  };
  return out;
};

/**
 * Z-score standardization for a matrix of numbers along the columns.
 */
export const standardizeNumbers = (data: number[][]) => {
  const total: number[] = Array(data[0].length).fill(0);
  const mean: number[] = Array(data[0].length).fill(0);

  for (const row of data) {
    for (let i = 0; i < row.length; ++i) {
      total[i] += row[i];
    }
  }

  for (let i = 0; i < total.length; ++i) {
    mean[i] = total[i] / data.length;
  }

  const std: number[] = Array(data[0].length).fill(0);

  for (const row of data) {
    for (let i = 0; i < row.length; ++i) {
      std[i] += (row[i] - mean[i]) ** 2;
    }
  }

  for (let i = 0; i < std.length; ++i) {
    std[i] = Math.sqrt(std[i] / data.length);
  }

  const standardized = [];
  for (const row of data) {
    const newRow = [];
    for (let i = 0; i < row.length; ++i) {
      newRow.push((row[i] - mean[i]) / std[i]);
    }
    standardized.push(newRow);
  }

  return { standardized, mean, std, total };
};

/**
 * Min-max normalization for a matrix of numbers along the columns.
 */
export const normalizeNumbers = (data: number[][]) => {
  const min: number[] = Array(data[0].length).fill(Infinity);
  const max: number[] = Array(data[0].length).fill(-Infinity);

  for (const row of data) {
    for (let i = 0; i < row.length; ++i) {
      if (row[i] < min[i]) {
        min[i] = row[i];
      }
      if (row[i] > max[i]) {
        max[i] = row[i];
      }
    }
  }

  const normalized = [];
  for (const row of data) {
    const newRow = [];
    for (let i = 0; i < row.length; ++i) {
      newRow.push((row[i] - min[i]) / (max[i] - min[i]));
    }
    normalized.push(newRow);
  }

  return { normalized, min, max };
};
