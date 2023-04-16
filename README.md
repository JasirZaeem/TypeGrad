# TypeGrad

TypeGrad is a simple scalar autograd library in TypeScript. It is designed to be easy to use and understand, and to be a good starting point for learning about autograd. Based on the micrograd library by [Andrej Karpathy](https://github.com/karpathy/micrograd)

Note: This library is not intended for production use. It is not optimized for speed, and it does not support GPU acceleration. Try [Shumai](https://github.com/facebookresearch/shumai) by Facebook Research or [TensorFlow.js](https://github.com/tensorflow/tfjs) for GPU acceleration and vectorization.

### Installation

```bash
npm install typegrad
```

### Usage

```typescript
import { v } from "typegrad";

const x = v(3, "x");
const y = v(4, "y");

// z = 4x^3 + 2y^2
let z = x
  .pow(3)
  .mul(v(4))
  .add(y.pow(2).mul(v(2)));

console.log(z.toString()); // Value(140.0 +)

z.backward(); // compute gradients for all variables with respect to z
z.printComputationGraph();

/*
Value(140.0 + grad: 1.0)
  Value(108.0 * grad: 1.0)
    Value(27.0 ^3 grad: 4.0)
      Value(3.0 (x) grad: 108.0)
    Value(4.0 grad: 27.0)
  Value(32.0 * grad: 1.0)
    Value(16.0 ^2 grad: 2.0)
      Value(4.0 (y) grad: 16.0)
    Value(2.0 grad: 16.0)
*/

console.log(`dz/dx at x=3: ${x.grad}`); // dz/dx at x=3: 108
console.log(`dz/dy at y=4: ${y.grad}`); // dz/dy at y=4: 16
```

There are also some implementations of feedforward neural networks in TypeGrad, based on composition of the available operations.

### ANN Example

Implementing http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html in TypeGrad

`data.ts`

```typescript
export const data = {
  x: [
    [2.104e3, 3.0],
    [1.6e3, 3.0],
    // ... 45 more rows
  ],
  y: [
    [3.999e5],
    [3.299e5],
    // ... 45 more rows
  ],
};
```

`index.ts`

```typescript
import * as tg from "typegrad";
import { data } from "./data.js";

// Z-score standardization
const {
  standardized: xStandardized,
  mean: xMean,
  std: xStd,
} = tg.standardizeNumbers(data.x);
const {
  standardized: yStandardized,
  mean: yMean,
  std: yStd,
} = tg.standardizeNumbers(data.y);

// Convert to TypeGrad Values
const x = tg.fromMatrix(xStandardized);
const y = tg.fromMatrix(yStandardized);

// A single neuron with 2 inputs and identity activation (y = x)
// const model = tg.neuron(2, tg.Activations.Identity);

// Or A single layer with 2 inputs, 1 output, and identity activation (y = x)
// const model = tg.layer(2, 1, tg.Activations.Identity);

// Or A Multi-Layer Perceptron with 2 inputs and 1 layer with 1 output with identity activation
// const model = tg.MLP(2, [[1, tg.Activations.Identity]]);

// Or A Sequential module with one linear layer with 2 inputs and 1 output with identity activation
const model = tg.sequential(tg.layer(2, 1, tg.Activations.Identity));

// Stochastic Gradient Descent with learning rate 0.1
const optimizer = tg.SGD({ lr: 0.1, model });

// Training loop for 100 iterations
for (let i = 0; i < 100; ++i) {
  // runBatch run an array of model inputs
  const yPreds = tg.runBatch(model, x);
  const loss = tg.meanSquaredError(tg.getValues(yPreds), tg.getValues(y));

  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();

  if (i % 10 === 0 || i === 99) {
    console.log(`Loss: ${loss.value}`);
  }
}

console.log("Predictions:");

const yPred = model.forward([
  new tg.Value((1650 - xMean[0]) / xStd[0]),
  new tg.Value((3 - xMean[1]) / xStd[1]),
]);

// yPred.value in case a single neuron was used, rest return an array of values
const yPredActual = yPred[0].value * yStd[0] + yMean[0];
console.log(`x: 1650, 3, y: 293081, yPred: ${yPredActual}`);
```

```bash
$ node index.js
Loss: 2.5167192548196677
Loss: 0.38943873428395853
Loss: 0.28577348426944876
Loss: 0.27001394631673464
Loss: 0.26752369202704895
Loss: 0.2671292383032242
Loss: 0.26706674630712984
Loss: 0.2670568457813185
Loss: 0.2670552772524149
Loss: 0.26705502875217013
Loss: 0.26705499088197726
Predictions:
x: 1650, 3, y: 293081, yPred: 293081.9612040153
```
