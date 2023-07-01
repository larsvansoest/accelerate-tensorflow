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



histogram :: Acc (Vector Int) -> Acc (Vector Int)
histogram xs =
  let zeros = fill (shape xs) 0
      ones  = fill (shape xs) 1
  in
  permute (+) zeros (\ix -> Just_ (I1 (xs!ix))) ones
 


-- Our benchmark harness.
main :: IO ()
main = defaultMain [
  bgroup "fib" [ bench "TF generateEven"  $ whnf (run @TensorFlow) generateEven
               , bench "Interpreter generateEven"  $ whnf (run @Interpreter) generateEven
               , bench "histogram" $ whnf (run @TensorFlow) (histogram (use (fromList (Z :. 20) [0,0,1,2,1,1,2,4,8,3,4,9,8,3,2,5,5,3,1,2] :: Vector Int)))
               , bench "histogram" $ whnf (run @Interpreter) (histogram (use (fromList (Z :. 20) [0,0,1,2,1,1,2,4,8,3,4,9,8,3,2,5,5,3,1,2] :: Vector Int)))
               ]
  ]
