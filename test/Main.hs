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
try = let zeros :: Acc (Vector Int) 
          zeros = fill (constant (Z:.10)) 0
          ones :: Acc (Vector Int) 
          ones = fill (constant (Z:.10)) 0
          in test @UniformScheduleFun @TensorFlowKernel $ permute (+) zeros Just_ ones
  
  
  -- let zeros :: Acc (Vector Int) 
  --         zeros = fill (constant (Z:.10)) 0
  --         ones :: Acc (Vector Int) 
  --         ones = fill (constant (Z:.10)) 0
  --         in test @UniformScheduleFun @TensorFlowKernel $ permute (+) zeros Just_ ones



--try = test @UniformScheduleFun @TensorFlowKernel $ scatter to to to

--try = test @UniformScheduleFun @TensorFlowKernel $ map @DIM1 @Int (+ 1 * 2) (use (fromList (Z :. 10) [0..]))

  -- test @UniformScheduleFun @TensorFlowKernel $ zipWith @DIM1 @Int (+) (use (fromList (Z :. 10) [0..])) (use (fromList (Z :. 10) [0..]))

 -- test @UniformScheduleFun @InterpretKernel $ zipWith @DIM1 @Int (+)