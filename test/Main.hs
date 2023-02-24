{-# LANGUAGE TypeApplications #-}
import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate
import Data.Array.Accelerate.Interpreter
import Prelude hiding (map, zipWith, (+))
import Data.Accelerate.TensorFlow.Kernel

main :: IO ()
main = putStrLn try

--x = (generate (I2 10 11) (\(I2 i j) -> i + j + 1))

try :: String
-- try = test @UniformScheduleFun @TensorKernel $ x
try = let zeros :: Acc (Matrix Int) 
          zeros = use $ fromList (Z :. 2 :. 3) [0..]
          ones :: Acc (Matrix Int) 
          ones = use $ fromList (Z :. 2 :. 3) [1..]
          in test @UniformScheduleFun @TensorKernel $ permute (+) zeros Just_ ones


  -- let zeros :: Acc (Vector Int) 
  --         zeros = fill (constant (Z:.10)) 0
  --         ones :: Acc (Vector Int) 
  --         ones = fill (constant (Z:.10)) 0
  --         in test @UniformScheduleFun @TensorFlowKernel $ permute (+) zeros Just_ ones



--try = test @UniformScheduleFun @TensorFlowKernel $ scatter to to to

--try = test @UniformScheduleFun @TensorKernel $ map @DIM1 @Int (\x -> (x + 1) * 2) (use (fromList (Z :. 10) [0..]))

  -- test @UniformScheduleFun @TensorFlowKernel $ zipWith @DIM1 @Int (+) (use (fromList (Z :. 10) [0..])) (use (fromList (Z :. 10) [0..]))

 -- test @UniformScheduleFun @InterpretKernel $ zipWith @DIM1 @Int (+)