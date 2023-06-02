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

-- The function we're benchmarking.
histogram :: Acc (Vector Int) -> Acc (Vector Int)
histogram xs =
  let zeros = fill (constant (Z:.10)) 0
      ones  = fill (shape xs)         1
  in
  permute (+) zeros (\ix -> Just_ (I1 (xs!ix))) ones

-- Our benchmark harness.
main :: IO ()
main = defaultMain [
  bgroup "fib" [ bench "TF histogram"  $ whnf (run @TensorFlow) $ histogram (use (fromList (Z :. 20) [0,0,1,2,1,1,2,4,8,3,4,9,8,3,2,5,5,3,1,2] :: Vector Int))
               , bench "Interpreter histogram"  $ whnf (run @Interpreter) $ histogram (use (fromList (Z :. 20) [0,0,1,2,1,1,2,4,8,3,4,9,8,3,2,5,5,3,1,2] :: Vector Int))
               ]
  ]
