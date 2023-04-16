import * as tg from "@/typegrad";

export class StochasticGradientDescent<
  T extends tg.Module | undefined
> extends tg.Optimizer {
  _lr: number;
  _model?: T;
  constructor(lr: number | tg.Value = 1e-3, model?: T) {
    super();
    this._model = model;
    if (typeof lr === "number") {
      this._lr = lr;
    } else {
      this._lr = lr.value;
    }
  }

  step(
    ...args: undefined extends T
      ? [arg: Iterable<{ value: number; grad: number }>]
      : [arg?: T]
  ) {
    const [parameters] = args;
    if (this._model) {
      for (const param of this._model.parameters) {
        param.value -= this._lr * param.grad;
      }
    } else if (tg.isIterable<{ value: number; grad: number }>(parameters)) {
      for (const param of parameters) {
        param.value -= this._lr * param.grad;
      }
    }
  }

  zeroGrad(
    ...args: undefined extends T
      ? [arg: Iterable<{ value: number; grad: number }>]
      : [arg?: T]
  ) {
    const [parameters] = args;
    if (this._model) {
      this._model.zeroGrad();
    } else if (tg.isIterable<{ value: number; grad: number }>(parameters)) {
      for (const param of parameters) {
        param.value = 0;
      }
    }
  }
}

export const SGD = <T extends tg.Module | undefined>({
  lr = 1e-3,
  model,
}: {
  lr?: number | tg.Value;
  model?: T;
} = {}) => new StochasticGradientDescent<T>(lr, model);
