{-# LANGUAGE TypeApplications #-}
import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate
import Data.Array.Accelerate.Interpreter
import Prelude hiding (map, zipWith)
import Data.Accelerate.TensorFlow.Operation (TensorFlowKernel)

main :: IO ()
main = putStrLn try

to :: Acc (Vector Int)
to = use (fromList (Z :. 10) [0..])

try :: String
try = let ab = fromList (Z :. 20) [0,0,1,2,1,1,2,4,8,3,4,9,8,3,2,5,5,3,1,2]
          histogram :: Acc (Vector Int) -> Acc (Vector Int)
          histogram xs =
            let zeros = fill (constant (Z:.10)) 0
                ones  = fill (shape xs)         1
            in
            permute (+) zeros (\ix -> Just_ (I1 (xs!ix))) ones
          in test @UniformScheduleFun @TensorFlowKernel $ histogram (use ab)
  

  
  --test @UniformScheduleFun @TensorFlowKernel $ scatter to to to
  
  -- test @UniformScheduleFun @TensorFlowKernel $ map @DIM1 @Int (+ 1 * 2) (use (fromList (Z :. 10) [0..]))

  -- test @UniformScheduleFun @TensorFlowKernel $ zipWith @DIM1 @Int (+) (use (fromList (Z :. 10) [0..])) (use (fromList (Z :. 10) [0..]))

 -- test @UniformScheduleFun @InterpretKernel $ zipWith @DIM1 @Int (+)