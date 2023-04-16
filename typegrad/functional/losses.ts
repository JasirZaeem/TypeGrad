import * as tg from "@/typegrad";

export const meanSquaredError = (
  yTrue: Iterable<tg.Value>,
  yPred: Iterable<tg.Value>
) => {
  const squaredErrors: tg.Value[] = [];
  const yti = yTrue[Symbol.iterator]();
  const ypi = yPred[Symbol.iterator]();
  let count = 0;
  for (
    let yt = yti.next(), yp = ypi.next();
    !yt.done && !yp.done;
    yt = yti.next(), yp = ypi.next()
  ) {
    squaredErrors.push(yp.value.sub(yt.value).pow(2));
    ++count;
  }

  return tg.sum(squaredErrors).div(new tg.Value(count));
};

export const MSE = meanSquaredError;
