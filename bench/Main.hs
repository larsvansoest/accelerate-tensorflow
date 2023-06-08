{-# LANGUAGE TypeApplications #-}
import Criterion.Main
import Data.Array.Accelerate
import Data.Accelerate.TensorFlow.Execute (TensorFlow)
import Data.Array.Accelerate.Interpreter (Interpreter)
import Data.Accelerate.TensorFlow.Operation
import Data.Accelerate.TensorFlow.Desugar
import Data.Accelerate.TensorFlow.Kernel
import Data.Array.Accelerate.Pretty.Schedule
import Data.Array.Accelerate.AST.Schedule.Sequential hiding (Exp)
import Data.Array.Accelerate.Pretty.Schedule.Sequential
import Prelude hiding (fromIntegral)

generateEven :: Acc (Array DIM1 Int64)
generateEven = generate (I1 5) (\(I1 i) -> fromIntegral $ i `mod` 2)

-- Our benchmark harness.
main :: IO ()
main = defaultMain [
  bgroup "fib" [ bench "TF generateEven"  $ whnf (run @TensorFlow) generateEven
               , bench "Interpreter generateEven"  $ whnf (run @Interpreter) generateEven
               ]
  ]
